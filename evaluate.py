"""
evaluate.py — Benchmark Evaluation on LIAR and FEVER Datasets
-------------------------------------------------------------
Evaluates the FactSense verification pipeline on samples from the
LIAR and FEVER benchmarks and reports:
  - Overall Accuracy
  - Macro-Averaged Precision, Recall, and F1
  - Per-class Metrics (SUPPORTED, REFUTED, INSUFFICIENT EVIDENCE)
  - Confusion Matrix
  - Latency per verification pass
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict

# Force UTF-8 on Windows console output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from detector import detect

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LIAR_TEST_PATH = os.path.join(DATA_DIR, "liar_test.tsv")

CLASSES = ["SUPPORTED", "REFUTED", "MISLEADING", "INSUFFICIENT_EVIDENCE"]

def map_liar_label(raw: str) -> str:
    r = raw.strip().lower()
    if r in ("true", "mostly-true"):
        return "SUPPORTED"
    elif r in ("half-true", "barely-true"):
        return "MISLEADING"
    elif r in ("false", "pants-fire"):
        return "REFUTED"
    else:
        return "INSUFFICIENT_EVIDENCE"

def map_fever_label(raw: str) -> str:
    r = raw.strip().upper()
    if "SUPPORT" in r:
        return "SUPPORTED"
    elif "REFUTE" in r:
        return "REFUTED"
    else:
        return "INSUFFICIENT_EVIDENCE"

def load_evaluation_data(num_samples: int = 30, dataset_choice: str = "all") -> List[Dict[str, str]]:
    samples = []

    # 1. FEVER samples
    if dataset_choice in ("all", "fever"):
        try:
            print("Loading FEVER evaluation split...")
            fever_df = pd.read_parquet("hf://datasets/pietrolesci/nli_fever/data/dev-00000-of-00001.parquet")
            target_per_class = max(2, (num_samples // 2) // 3) if dataset_choice == "all" else max(3, num_samples // 3)
            for lbl in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
                subset = fever_df[fever_df["fever_gold_label"] == lbl].head(target_per_class)
                for _, row in subset.iterrows():
                    claim = str(row.get("hypothesis", "")).strip()
                    if claim:
                        samples.append({
                            "claim": claim,
                            "gold": map_fever_label(lbl),
                            "dataset": "FEVER"
                        })
            print(f"Loaded {len(samples)} samples from FEVER.")
        except Exception as e:
            print(f"Notice: Could not load FEVER split ({e}).")

    # 2. LIAR samples
    if dataset_choice in ("all", "liar") and os.path.exists(LIAR_TEST_PATH):
        try:
            print("Loading LIAR evaluation split...")
            liar_items = []
            with open(LIAR_TEST_PATH, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 4:
                        raw_lbl = row[2].strip()
                        stmt = row[3].strip()
                        mapped = map_liar_label(raw_lbl)
                        if stmt and len(stmt) > 10:
                            liar_items.append({
                                "claim": stmt,
                                "gold": mapped,
                                "dataset": "LIAR"
                            })

            needed_liar = (num_samples - len(samples)) if dataset_choice == "all" else num_samples
            # Balance classes across LIAR
            by_class = defaultdict(list)
            for item in liar_items:
                by_class[item["gold"]].append(item)

            per_class_quota = max(2, needed_liar // 3)
            for c in CLASSES:
                samples.extend(by_class[c][:per_class_quota])

            print(f"Loaded {len(samples)} total samples.")
        except Exception as e:
            print(f"Notice: Could not load LIAR split ({e}).")

    return samples[:num_samples]

def compute_metrics(y_true: List[str], y_pred: List[str]):
    matrix = {c_true: {c_pred: 0 for c_pred in CLASSES} for c_true in CLASSES}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1

    total = len(y_true)
    correct = sum(matrix[c][c] for c in CLASSES)
    accuracy = correct / total if total > 0 else 0.0

    class_metrics = {}
    f1_list = []
    prec_list = []
    rec_list = []

    for c in CLASSES:
        tp = matrix[c][c]
        fp = sum(matrix[other][c] for other in CLASSES if other != c)
        fn = sum(matrix[c][other] for other in CLASSES if other != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_metrics[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)

    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))

    return {
        "accuracy": accuracy,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": matrix,
        "total": total,
    }

def run_evaluation(num_samples: int = 24, dataset_choice: str = "all"):
    print("\n" + "=" * 65)
    print(f"  FactSense Evaluation Benchmark (Dataset: {dataset_choice.upper()})")
    print(f"  Evaluating {num_samples} balanced claim-verification samples...")
    print("=" * 65)

    samples = load_evaluation_data(num_samples=num_samples, dataset_choice=dataset_choice)
    if not samples:
        print("Error: No evaluation samples could be loaded.")
        return

    y_true = []
    y_pred = []
    latencies = []

    print("\nRunning inference pipeline...")
    for idx, item in enumerate(samples, 1):
        claim = item["claim"]
        gold = item["gold"]
        dataset = item["dataset"]

        t0 = time.time()
        try:
            res = detect(claim)
            pred = res.get("verdict", "INSUFFICIENT_EVIDENCE")
            if pred == "INSUFFICIENT EVIDENCE":
                pred = "INSUFFICIENT_EVIDENCE"
            conf = res.get("confidence", 0.0)
            elapsed = res.get("elapsed_sec", time.time() - t0)
            
            # Print framing and stance summary if available
            framing = res.get("framing")
            if framing and framing.get("has_headline"):
                print(f"      [Framing] Severity: {framing.get('severity')} | Issues: {framing.get('framing_issues')}")
        except Exception as e:
            pred = "INSUFFICIENT_EVIDENCE"
            conf = 0.0
            elapsed = time.time() - t0

        latencies.append(elapsed)
        y_true.append(gold)
        y_pred.append(pred)

        match_symbol = "[OK]  " if gold == pred else "[DIFF]"
        print(f"[{idx:02d}/{len(samples):02d}] {match_symbol} Pred: {pred:<21} | Gold: {gold:<21} | {elapsed:.2f}s | {claim[:50]}...")

    results = compute_metrics(y_true, y_pred)
    avg_latency = float(np.mean(latencies))

    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    print(f"Total Evaluated Claims: {results['total']}")
    print(f"Overall Accuracy:       {results['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:        {results['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:           {results['macro_recall'] * 100:.2f}%")
    print(f"Macro F1 Score:         {results['macro_f1'] * 100:.2f}%")
    print(f"Average CPU Latency:    {avg_latency:.2f}s per claim")

    print("\nPer-Class Detailed Performance:")
    print(f"{'Class':<23} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    for c in CLASSES:
        m = results["class_metrics"][c]
        print(f"{c:<23} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}% | {m['f1']*100:>8.2f}% | {m['support']:>7}")

    print("\nConfusion Matrix (Row: True, Column: Predicted):")
    print(f"{'':<23} | {'SUPPORTED':<10} | {'REFUTED':<10} | {'MISLEADING':<10} | {'INSUFFICIENT'}")
    print("-" * 75)
    for r in CLASSES:
        row_str = f"{r:<23} | "
        for col in CLASSES:
            row_str += f"{results['confusion_matrix'][r][col]:>10} "
        print(row_str)
    print("=" * 75 + "\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FactSense benchmark evaluation.")
    parser.add_argument("--samples", type=int, default=15, help="Number of benchmark samples (default: 15).")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "fever", "liar"], help="Dataset to evaluate.")
    args = parser.parse_args()

    run_evaluation(num_samples=args.samples, dataset_choice=args.dataset)
