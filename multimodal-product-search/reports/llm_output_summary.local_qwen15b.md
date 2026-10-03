# Stage 5 — LLM Output Summary

- LLM/backend: Qwen/Qwen2.5-1.5B-Instruct
- Answers generated: 20
- JSON validity after repair/fallback: 100% (20/20)
- Schema-valid outputs: 100% (20/20)
- Direct-valid model outputs: 2
- Repaired outputs: 0
- Fallback outputs: 18
- Top products exposed to the model: 8

## Prompting template

The prompt provides query text/type, whether an image was supplied, only the top reranked catalog products, allowed labels, the required JSON schema, and strict grounding instructions.

## Example final answers

```json
{
  "query_id": "q001",
  "interpreted_need": {
    "category": "SOFA",
    "use_case": "",
    "positive_preferences": [
      "gray fabric sofa for a small living room"
    ],
    "negative_constraints": [
      "not leather"
    ],
    "visual_preferences": [],
    "uncertain_fields": []
  },
  "product_judgements": [
    {
      "product_id": "B086B5MRFB|amazon.in",
      "role": "substitute",
      "evidence": [
        "SOFA",
        "Grey",
        "Fabric",
        "Modern"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07B4MRLT8|amazon.com",
      "role": "substitute",
      "evidence": [
        "FURNITURE_COVER",
        "Grey",
        "stone"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07QJXW4JR|amazon.com",
      "role": "substitute",
      "evidence": [
        "SOFA",
        "Flax"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07B4D88DY|amazon.com",
      "role": "substitute",
      "evidence": [
        "SOFA",
        "Charcoal Grey",
        "Metal",
        "Industrial"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07K2JGTDN|amazon.com",
      "role": "substitute",
      "evidence": [
        "SOFA",
        "Marble (Light Grey)",
        "Fredrickson Marble"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    }
  ],
  "decision": "recommend_substitute",
  "customer_response": "The first result is the closest grounded candidate from this search set; compare its listed type, material, color, and style details with your request before choosing."
}
```
```json
{
  "query_id": "q002",
  "interpreted_need": {
    "category": "TABLE",
    "use_case": "",
    "positive_preferences": [
      "modern wooden coffee table with storage"
    ],
    "negative_constraints": [],
    "visual_preferences": [],
    "uncertain_fields": []
  },
  "product_judgements": [
    {
      "product_id": "B07DBF3VJY|amazon.com",
      "role": "substitute",
      "evidence": [
        "TABLE",
        "White",
        "wood",
        "Bold eclectic"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07K7NNS6N|amazon.co.uk",
      "role": "substitute",
      "evidence": [
        "TABLE",
        "Wild Oak",
        "Oak",
        "Modern"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07J1YW3YT|amazon.com",
      "role": "substitute",
      "evidence": [
        "TABLE",
        "Natural",
        "Wood",
        "Transitional"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07QC861L7|amazon.com",
      "role": "substitute",
      "evidence": [
        "TABLE",
        "Walnut/White",
        "Engineered Wood",
        "Coffee Table"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07SSCJKLN|amazon.in",
      "role": "substitute",
      "evidence": [
        "TABLE",
        "Espresso",
        "Engineered Wood",
        "Contemporary"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    }
  ],
  "decision": "recommend_substitute",
  "customer_response": "The first result is the closest grounded candidate from this search set; compare its listed type, material, color, and style details with your request before choosing."
}
```
```json
{
  "query_id": "q003",
  "interpreted_need": {
    "category": "HOME_FURNITURE_AND_DECOR",
    "use_case": "",
    "positive_preferences": [
      "black office chair with armrests"
    ],
    "negative_constraints": [
      "not a dining chair"
    ],
    "visual_preferences": [],
    "uncertain_fields": []
  },
  "product_judgements": [
    {
      "product_id": "B07SSHYD2Z|amazon.co.uk",
      "role": "substitute",
      "evidence": [
        "HOME_FURNITURE_AND_DECOR"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B081HX9SRM|amazon.co.uk",
      "role": "substitute",
      "evidence": [
        "ACCESSORY_OR_PART_OR_SUPPLY",
        "Black",
        "Alloy Steel"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B081HWKWN6|amazon.co.uk",
      "role": "substitute",
      "evidence": [
        "ACCESSORY_OR_PART_OR_SUPPLY",
        "Black",
        "Alloy Steel"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07QLGRVPN|amazon.in",
      "role": "substitute",
      "evidence": [
        "CHAIR",
        "Black",
        "Plastic",
        "Mid-Back Mesh Chair"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    },
    {
      "product_id": "B07HMPPZRN|amazon.com",
      "role": "substitute",
      "evidence": [
        "CHAIR",
        "Black",
        "Steel",
        "Foldable Rocking Chair"
      ],
      "constraint_violations": [],
      "reason": "This candidate appears near the top of the grounded retrieval results and has matching catalog metadata worth comparing with the request."
    }
  ],
  "decision": "recommend_substitute",
  "customer_response": "The first result is the closest grounded candidate from this search set; compare its listed type, material, color, and style details with your request before choosing."
}
```

## Limitations

The language model does not perform retrieval and cannot inspect omitted candidates. A text-only catalog description may not expose every visual attribute. Invalid or unsupported generations are rejected; deterministic fallback output is conservative rather than pretending certainty.
