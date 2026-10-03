"""Stage 5: grounded structured answers from a small local instruction model.

Retrieval is never delegated to the LLM. The model receives only the top reranked
products. Invalid output is retried and then replaced by a deterministic grounded
fallback so the final JSONL always satisfies the required schema.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_parent, load_jsonl, safe_text, write_jsonl

ALLOWED_ROLES = {"exact", "substitute", "irrelevant"}
ALLOWED_DECISIONS = {
    "recommend_exact", "recommend_exact_with_warning", "recommend_substitute",
    "no_good_match", "ask_clarification",
}
FORBIDDEN_RESPONSE_TERMS = {"price", "stock", "review", "rating", "shipping", "discount"}


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0; in_string = False; escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape: escape = False
            elif ch == "\\": escape = True
            elif ch == '"': in_string = False
        else:
            if ch == '"': in_string = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj = json.loads(text[start:i+1])
                    if not isinstance(obj, dict):
                        raise ValueError("JSON root is not an object")
                    return obj
    raise ValueError("unterminated JSON object")


def validate_answer(obj: dict[str, Any], *, qid: str, allowed_ids: set[str]) -> None:
    required = {"query_id", "interpreted_need", "product_judgements", "decision", "customer_response"}
    if not required.issubset(obj):
        raise ValueError(f"missing keys: {sorted(required - set(obj))}")
    if obj["query_id"] != qid:
        raise ValueError("query_id mismatch")
    need = obj["interpreted_need"]
    if not isinstance(need, dict) or not isinstance(need.get("category"), str):
        raise ValueError("interpreted_need.category must be a string")
    for field in ["positive_preferences", "negative_constraints", "visual_preferences", "uncertain_fields"]:
        if not isinstance(need.get(field), list):
            raise ValueError(f"interpreted_need.{field} must be a list")
    judgements = obj["product_judgements"]
    if not isinstance(judgements, list) or not judgements:
        raise ValueError("product_judgements must be a non-empty list")
    for j in judgements:
        if j.get("product_id") not in allowed_ids:
            raise ValueError(f"product_id outside reranked top-k: {j.get('product_id')}")
        if j.get("role") not in ALLOWED_ROLES:
            raise ValueError(f"invalid role: {j.get('role')}")
        if not isinstance(j.get("evidence"), list):
            raise ValueError("evidence must be a list")
        if not isinstance(j.get("constraint_violations"), list):
            raise ValueError("constraint_violations must be a list")
        if not isinstance(j.get("reason"), str) or not j["reason"].strip():
            raise ValueError("judgement reason is missing")
    if obj["decision"] not in ALLOWED_DECISIONS:
        raise ValueError(f"invalid decision: {obj['decision']}")
    response = obj["customer_response"]
    if not isinstance(response, str) or len(response.split()) < 8:
        raise ValueError("customer_response is too short")
    low = response.lower()
    used = sorted(term for term in FORBIDDEN_RESPONSE_TERMS if term in low)
    if used:
        raise ValueError(f"customer_response contains unavailable-commerce terms: {used}")


def infer_need(query: dict, top_rows: list[Any]) -> dict[str, Any]:
    text = (query.get("query_text") or "").strip()
    negative = []
    for match in re.finditer(r"\bnot\s+([^,.;]+)", text, flags=re.I):
        phrase = "not " + match.group(1).strip()
        if phrase not in negative:
            negative.append(phrase)
    category = ""
    if top_rows:
        category = safe_text(getattr(top_rows[0], "product_type", "")).strip()
    if not category:
        category = "visual match" if query["query_type"] == "image" else "product"
    positive_text = re.sub(r"\bnot\s+[^,.;]+", "", text, flags=re.I)
    prefs = [p.strip() for p in re.split(r",|\band\b", positive_text) if p.strip()]
    return {
        "category": category,
        "use_case": "",
        "positive_preferences": prefs[:6],
        "negative_constraints": negative[:6],
        "visual_preferences": ["similar to the provided image"] if query["query_type"] in {"image", "image_text"} else [],
        "uncertain_fields": [],
    }


def fallback_answer(query: dict, top_hits: list[dict], cat_by_id: dict[str, Any]) -> dict[str, Any]:
    selected = top_hits[:min(5, len(top_hits))]
    rows = [cat_by_id[h["product_id"]] for h in selected]
    need = infer_need(query, rows)
    judgements = []
    for rank, h in enumerate(selected):
        row = cat_by_id[h["product_id"]]
        evidence = []
        for field in ("product_type", "color", "material", "style"):
            value = safe_text(getattr(row, field, "")).strip()
            if value and value not in evidence:
                evidence.append(value)
        role = "substitute"
        reason = "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
        if rank == 0 and query["query_type"] == "image":
            reason = "This is the highest-ranked visual candidate from the multimodal retrieval results; the catalog metadata should be checked against any unstated preferences."
        judgements.append({
            "product_id": h["product_id"],
            "role": role,
            "evidence": evidence[:5] or [safe_text(getattr(row, "title", "")).strip()],
            "constraint_violations": [],
            "reason": reason,
        })
    return {
        "query_id": query["query_id"],
        "interpreted_need": need,
        "product_judgements": judgements,
        "decision": "recommend_substitute" if judgements else "no_good_match",
        "customer_response": "The first result is the closest grounded candidate from this search set; compare its listed type, material, color, and style details with your request before choosing.",
    }


def build_prompt(query: dict, hits: list[dict], cat_by_id: dict[str, Any]) -> str:
    products = []
    for h in hits:
        r = cat_by_id[h["product_id"]]
        products.append({
            "product_id": h["product_id"],
            "title": safe_text(getattr(r, "title", "")),
            "product_type": safe_text(getattr(r, "product_type", "")),
            "category": safe_text(getattr(r, "category_path", "")),
            "brand": safe_text(getattr(r, "brand", "")),
            "color": safe_text(getattr(r, "color", "")),
            "material": safe_text(getattr(r, "material", "")),
            "style": safe_text(getattr(r, "style", "")),
            "description": safe_text(getattr(r, "description", ""))[:1000],
        })
    schema = {
        "query_id": query["query_id"],
        "interpreted_need": {
            "category": "string", "use_case": "string",
            "positive_preferences": [], "negative_constraints": [],
            "visual_preferences": [], "uncertain_fields": [],
        },
        "product_judgements": [{
            "product_id": "must be one of the supplied IDs", "role": "exact|substitute|irrelevant",
            "evidence": [], "constraint_violations": [], "reason": "string",
        }],
        "decision": "recommend_exact|recommend_exact_with_warning|recommend_substitute|no_good_match|ask_clarification",
        "customer_response": "clear grounded response",
    }
    return f"""You are a product search assistant. Return ONLY valid JSON, with no markdown.
Use only the product metadata provided. Do not invent unsupported attributes or commerce facts.
Respect explicit negative constraints. Judge only supplied products and do not introduce other product IDs.

Query ID: {query['query_id']}
Query type: {query['query_type']}
Query text: {query.get('query_text') or '[none]'}
Query image provided: {'yes' if query.get('query_image_path') else 'no'}

Allowed roles: exact, substitute, irrelevant
Allowed decisions: recommend_exact, recommend_exact_with_warning, recommend_substitute, no_good_match, ask_clarification
Required schema example: {json.dumps(schema, ensure_ascii=False)}

Products: {json.dumps(products, ensure_ascii=False)}
"""


class LocalGenerator:
    def __init__(self, model_name: str, load_in_4bit: bool, adapter_path: str | None,
                 max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:
                print(f"4-bit configuration unavailable ({exc}); loading without quantization")
                kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def __call__(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Return valid JSON only. Stay grounded in supplied metadata."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = messages[0]["content"] + "\n\n" + messages[1]["content"]
        enc = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--llm_backend", choices=["local", "fallback"], default="local")
    ap.add_argument("--llm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter_path")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1200)
    ap.add_argument("--local_json_repair_attempts", type=int, default=2)
    ap.add_argument("--out", default="outputs/final_answers.jsonl")
    ap.add_argument("--summary", default="reports/llm_output_summary.md")
    ap.add_argument("--debug_raw_out", default="reports/raw_llm_debug.jsonl")
    args = ap.parse_args()

    catalog = pd.read_parquet(args.catalog)
    cat_by_id = {str(r.product_id): r for r in catalog.itertuples()}
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    reranked = {r["query_id"]: r for r in load_jsonl(args.reranked)}

    generator = None
    if args.llm_backend == "local":
        generator = LocalGenerator(args.llm_model, args.load_in_4bit, args.adapter_path, args.max_new_tokens)

    answers = []; debug = []
    direct_valid = repaired = fallbacks = 0
    for qid, q in queries.items():
        hits = reranked.get(qid, {}).get("results", [])[:args.top_k]
        if not hits:
            raise ValueError(f"{qid} has no reranked top-{args.top_k} candidates")
        allowed_ids = {h["product_id"] for h in hits}
        prompt = build_prompt(q, hits, cat_by_id)
        accepted = None; raw_attempts = []; error = ""; accepted_via = ""
        if generator is not None:
            for attempt in range(args.local_json_repair_attempts + 1):
                use_prompt = prompt if attempt == 0 else (
                    prompt + "\n\nYour previous output was invalid. Error: " + error +
                    "\nReturn a corrected JSON object only. Previous output:\n" + raw_attempts[-1]
                )
                raw = generator(use_prompt)
                raw_attempts.append(raw)
                try:
                    obj = extract_json(raw)
                    validate_answer(obj, qid=qid, allowed_ids=allowed_ids)
                    accepted = obj
                    if attempt == 0:
                        direct_valid += 1
                        accepted_via = "direct"
                    else:
                        repaired += 1
                        accepted_via = "repair"
                    break
                except Exception as exc:
                    error = str(exc)
        if accepted is None:
            accepted = fallback_answer(q, hits, cat_by_id)
            validate_answer(accepted, qid=qid, allowed_ids=allowed_ids)
            fallbacks += 1
            accepted_via = "fallback"
        answers.append(accepted)
        debug.append({
            "query_id": qid,
            "direct_valid": accepted_via == "direct",
            "attempts": len(raw_attempts),
            "accepted_via": accepted_via,
            "last_error": error,
            "raw_outputs": raw_attempts,
        })

    write_jsonl(args.out, answers)
    write_jsonl(args.debug_raw_out, debug)
    total = len(answers)
    model_label = args.llm_model if args.llm_backend == "local" else "deterministic grounded fallback"
    examples = [json.dumps(a, ensure_ascii=False, indent=2) for a in answers[:3]]
    ensure_parent(args.summary)
    Path(args.summary).write_text("\n".join([
        "# Stage 5 — LLM Output Summary", "",
        f"- LLM/backend: {model_label}",
        f"- Answers generated: {total}",
        f"- JSON validity after repair/fallback: 100% ({total}/{total})",
        f"- Schema-valid outputs: 100% ({total}/{total})",
        f"- Direct-valid model outputs: {direct_valid}",
        f"- Repaired outputs: {repaired}",
        f"- Fallback outputs: {fallbacks}",
        f"- Top products exposed to the model: {args.top_k}",
        "", "## Prompting template", "",
        "The prompt provides query text/type, whether an image was supplied, only the top reranked catalog products, allowed labels, the required JSON schema, and strict grounding instructions.",
        "", "## Example final answers", "",
        *[f"```json\n{x}\n```" for x in examples],
        "", "## Limitations", "",
        "The language model does not perform retrieval and cannot inspect omitted candidates. A text-only catalog description may not expose every visual attribute. Invalid or unsupported generations are rejected; deterministic fallback output is conservative rather than pretending certainty.",
    ]) + "\n", encoding="utf-8")
    print(f"wrote {total} schema-valid answers to {args.out}")


if __name__ == "__main__":
    main()
