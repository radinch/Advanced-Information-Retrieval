import math
from typing import List


class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError('actual and predicted must have the same length')

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Macro precision over queries.
        """
        self._validate(actual, predicted)
        if not predicted:
            return 0.0
        precisions = []
        for rel, pred in zip(actual, predicted):
            pred_set = set(pred)
            if not pred_set:
                precisions.append(0.0)
            else:
                precisions.append(len(set(rel) & pred_set) / len(pred_set))
        return sum(precisions) / len(precisions) if precisions else 0.0

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Macro recall over queries.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0
        recalls = []
        for rel, pred in zip(actual, predicted):
            rel_set = set(rel)
            if not rel_set:
                recalls.append(0.0)
            else:
                recalls.append(len(rel_set & set(pred)) / len(rel_set))
        return sum(recalls) / len(recalls) if recalls else 0.0

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        F1 computed from macro precision and macro recall.
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        rel_set = set(actual)
        if not rel_set:
            return 0.0
        hits = 0
        precision_sum = 0.0
        seen = set()
        for rank, doc_id in enumerate(predicted, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            if doc_id in rel_set:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / len(rel_set)

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Mean AP across all queries. Kept for compatibility with the skeleton.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0
        return sum(self._average_precision_single(a, p) for a, p in zip(actual, predicted)) / len(actual)

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Mean Average Precision.
        """
        return self.calculate_AP(actual, predicted)

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        rel_set = set(actual)
        dcg = 0.0
        for rank, doc_id in enumerate(predicted, start=1):
            relevance = 1 if doc_id in rel_set else 0
            if relevance:
                dcg += relevance / math.log2(rank + 1)
        return dcg

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Mean DCG. Original misspelled method name retained.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0
        return sum(self._dcg_single(a, p) for a, p in zip(actual, predicted)) / len(actual)

    def calculate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        return self.cacluate_DCG(actual, predicted)

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Mean NDCG with binary relevance.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0
        ndcgs = []
        for rel, pred in zip(actual, predicted):
            dcg = self._dcg_single(rel, pred)
            ideal_len = min(len(set(rel)), len(pred)) if pred else len(set(rel))
            ideal_pred = list(dict.fromkeys(rel))[:ideal_len]
            idcg = self._dcg_single(rel, ideal_pred)
            ndcgs.append(0.0 if idcg == 0 else dcg / idcg)
        return sum(ndcgs) / len(ndcgs)

    def calculate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        return self.cacluate_NDCG(actual, predicted)

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Mean reciprocal rank. Original misspelled method name retained.
        """
        self._validate(actual, predicted)
        if not actual:
            return 0.0
        reciprocal_ranks = []
        for rel, pred in zip(actual, predicted):
            rel_set = set(rel)
            rr = 0.0
            for rank, doc_id in enumerate(pred, start=1):
                if doc_id in rel_set:
                    rr = 1 / rank
                    break
            reciprocal_ranks.append(rr)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    def calculate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        return self.cacluate_RR(actual, predicted)

    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Mean reciprocal rank.
        """
        return self.cacluate_RR(actual, predicted)

    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        return self.cacluate_MRR(actual, predicted)

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Print the evaluation metrics.
        """
        print(f'name = {self.name}')
        print(f'Precision = {precision:.6f}')
        print(f'Recall = {recall:.6f}')
        print(f'F1 = {f1:.6f}')
        print(f'AP = {ap:.6f}')
        print(f'MAP = {map:.6f}')
        print(f'DCG = {dcg:.6f}')
        print(f'NDCG = {ndcg:.6f}')
        print(f'RR = {rr:.6f}')
        print(f'MRR = {mrr:.6f}')

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Log evaluation metrics to Weights & Biases if installed.
        """
        try:
            import wandb
            wandb.init(project='MIR-2026', name=self.name)
            wandb.log({
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'AP': ap,
                'MAP': map,
                'DCG': dcg,
                'NDCG': ndcg,
                'RR': rr,
                'MRR': mrr,
            })
            wandb.finish()
        except Exception as exc:
            print(f'Could not log to wandb: {exc}')

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Convenience method returning all metrics as a dictionary.
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'map': map_score,
            'dcg': dcg,
            'ndcg': ndcg,
            'rr': rr,
            'mrr': mrr,
        }
