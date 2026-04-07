"""Evaluate the Smart Meeting Assistant benchmark datasets.

This script uses the existing local services to score:
- summarization
- action items
- sentiment and engagement
- translation structure preservation

It is intentionally lightweight so it can run inside the course project
environment without extra evaluation libraries.
"""

import argparse
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.action_items_service import get_action_items_service
from backend.services.sentiment_service import get_sentiment_service
from backend.services.summarization_service import get_summarization_service
from backend.services.translation_service import get_translation_service


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "smart_meeting_benchmark_v1"


def normalize_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s\u4e00-\u9fff]", "", value)
    return value.strip()


def text_units(text: Any) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if " " in normalized:
        return normalized.split()
    return list(normalized)


def token_f1(a: Any, b: Any) -> float:
    units_a = text_units(a)
    units_b = text_units(b)
    if not units_a or not units_b:
        return 0.0

    set_a = {}
    for item in units_a:
        set_a[item] = set_a.get(item, 0) + 1
    set_b = {}
    for item in units_b:
        set_b[item] = set_b.get(item, 0) + 1

    overlap = 0
    for item, count in set_a.items():
        overlap += min(count, set_b.get(item, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(units_a)
    recall = overlap / len(units_b)
    return 2 * precision * recall / (precision + recall)


def greedy_match_count(pred_items: Sequence[str], ref_items: Sequence[str], threshold: float = 0.5) -> Tuple[int, int, int]:
    used = set()
    tp = 0
    for pred in pred_items:
        best_idx = None
        best_score = 0.0
        for idx, ref in enumerate(ref_items):
            if idx in used:
                continue
            score = token_f1(pred, ref)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= threshold:
            used.add(best_idx)
            tp += 1

    fp = max(0, len(pred_items) - tp)
    fn = max(0, len(ref_items) - tp)
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def build_transcript(case: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    segments = []
    lines = []
    for idx, seg in enumerate(case.get("transcript_segments", [])):
        start = float(seg.get("start", idx * 5.0))
        end = float(seg.get("end", start + 3.0))
        speaker = str(seg.get("speaker", "Speaker 1")).strip() or "Speaker 1"
        text = str(seg.get("text", "")).strip()
        record = {
            "start": start,
            "end": end,
            "speaker": speaker,
            "text": text,
        }
        segments.append(record)
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines), segments


def load_cases(dataset_dir: Path, filename: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    payload = json.loads((dataset_dir / filename).read_text())
    cases = payload.get("cases", [])
    if split:
        cases = [case for case in cases if case.get("split") == split]
    return cases


def evaluate_summarization(dataset_dir: Path, split: Optional[str]) -> Dict[str, Any]:
    service = get_summarization_service()
    cases = load_cases(dataset_dir, "dataset_summarization.json", split)
    totals = {
        "key_topics": [0, 0, 0],
        "decisions": [0, 0, 0],
        "blockers": [0, 0, 0],
        "next_steps": [0, 0, 0],
    }
    per_case = []

    for case in cases:
        transcript, segments = build_transcript(case)
        predicted = service.summarize(transcript, segments)
        reference = case["light_reference"]["reference_summary"]
        case_result = {"id": case["id"]}

        for field in totals:
            pred_items = [str(x) for x in predicted.get(field, []) or []]
            ref_items = [str(x) for x in reference.get(field, []) or []]
            tp, fp, fn = greedy_match_count(pred_items, ref_items, threshold=0.45)
            totals[field][0] += tp
            totals[field][1] += fp
            totals[field][2] += fn
            case_result[field] = precision_recall_f1(tp, fp, fn)

        per_case.append(case_result)

    summary = {
        field: precision_recall_f1(values[0], values[1], values[2])
        for field, values in totals.items()
    }
    return {
        "num_cases": len(cases),
        "summary": summary,
        "cases": per_case,
    }


def normalize_action_item(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "assignee": normalize_text(item.get("assignee", "unknown")),
        "task": normalize_text(item.get("task", "")),
        "deadline": normalize_text(item.get("deadline", "")),
        "source_text": normalize_text(item.get("source_text", "")),
    }


def action_item_match_score(pred: Dict[str, str], ref: Dict[str, str]) -> float:
    assignee_score = 1.0 if pred["assignee"] == ref["assignee"] else 0.0
    if ref["assignee"] == "unknown" and pred["assignee"] in {"", "unknown"}:
        assignee_score = 1.0
    task_score = token_f1(pred["task"], ref["task"])
    deadline_score = 1.0 if (not ref["deadline"] and not pred["deadline"]) or pred["deadline"] == ref["deadline"] else 0.0
    return 0.5 * task_score + 0.35 * assignee_score + 0.15 * deadline_score


def evaluate_action_items(dataset_dir: Path, split: Optional[str]) -> Dict[str, Any]:
    service = get_action_items_service()
    cases = load_cases(dataset_dir, "dataset_action_items.json", split)

    tp = fp = fn = 0
    assignee_correct = 0
    deadline_correct = 0
    matched_items = 0
    per_case = []

    for case in cases:
        transcript, segments = build_transcript(case)
        predicted_items = [normalize_action_item(item) for item in service.extract(transcript, segments)]
        ref_items = [normalize_action_item(item) for item in case["light_reference"]["reference_action_items"]]
        used = set()
        case_tp = 0
        for pred in predicted_items:
            best_idx = None
            best_score = 0.0
            for idx, ref in enumerate(ref_items):
                if idx in used:
                    continue
                score = action_item_match_score(pred, ref)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is not None and best_score >= 0.65:
                used.add(best_idx)
                case_tp += 1
                matched_items += 1
                if pred["assignee"] == ref_items[best_idx]["assignee"]:
                    assignee_correct += 1
                if pred["deadline"] == ref_items[best_idx]["deadline"]:
                    deadline_correct += 1

        case_fp = max(0, len(predicted_items) - case_tp)
        case_fn = max(0, len(ref_items) - case_tp)
        tp += case_tp
        fp += case_fp
        fn += case_fn
        per_case.append({
            "id": case["id"],
            "matches": case_tp,
            "predicted": len(predicted_items),
            "reference": len(ref_items),
            **precision_recall_f1(case_tp, case_fp, case_fn),
        })

    metrics = precision_recall_f1(tp, fp, fn)
    metrics["assignee_accuracy"] = round(assignee_correct / matched_items, 4) if matched_items else 0.0
    metrics["deadline_accuracy"] = round(deadline_correct / matched_items, 4) if matched_items else 0.0

    return {
        "num_cases": len(cases),
        "summary": metrics,
        "cases": per_case,
    }


def signal_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    parts = []
    for key in ("statement", "signal", "topic", "description", "evidence"):
        value = item.get(key)
        if value:
            parts.append(str(value))
    return " ".join(parts)


def evaluate_sentiment(dataset_dir: Path, split: Optional[str]) -> Dict[str, Any]:
    service = get_sentiment_service()
    cases = load_cases(dataset_dir, "dataset_sentiment_engagement.json", split)
    overall_correct = 0
    engagement_correct = 0
    totals = {
        "agreements": [0, 0, 0],
        "disagreements": [0, 0, 0],
        "hesitation_signals": [0, 0, 0],
        "tension_points": [0, 0, 0],
    }
    per_case = []

    for case in cases:
        transcript, segments = build_transcript(case)
        predicted = service.analyze(transcript, segments)
        reference = case["light_reference"]["reference_sentiment"]

        if normalize_text(predicted.get("overall_sentiment")) == normalize_text(reference.get("overall_sentiment")):
            overall_correct += 1
        if normalize_text(predicted.get("engagement_level")) == normalize_text(reference.get("engagement_level")):
            engagement_correct += 1

        case_result = {
            "id": case["id"],
            "overall_sentiment_correct": normalize_text(predicted.get("overall_sentiment")) == normalize_text(reference.get("overall_sentiment")),
            "engagement_level_correct": normalize_text(predicted.get("engagement_level")) == normalize_text(reference.get("engagement_level")),
        }

        for field in totals:
            pred_items = [signal_to_text(item) for item in predicted.get(field, []) or []]
            ref_items = [signal_to_text(item) for item in reference.get(field, []) or []]
            tp, fp, fn = greedy_match_count(pred_items, ref_items, threshold=0.35)
            totals[field][0] += tp
            totals[field][1] += fp
            totals[field][2] += fn
            case_result[field] = precision_recall_f1(tp, fp, fn)

        per_case.append(case_result)

    summary = {
        "overall_sentiment_accuracy": round(overall_correct / len(cases), 4) if cases else 0.0,
        "engagement_level_accuracy": round(engagement_correct / len(cases), 4) if cases else 0.0,
    }
    for field, values in totals.items():
        summary[field] = precision_recall_f1(values[0], values[1], values[2])

    return {
        "num_cases": len(cases),
        "summary": summary,
        "cases": per_case,
    }


def format_translated_lines(segments: Sequence[Dict[str, Any]]) -> List[str]:
    lines = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text_translated") or seg.get("text") or ""
        lines.append(f"[{start:.1f}] {speaker}: {text}")
    return lines


def evaluate_translation(dataset_dir: Path, split: Optional[str]) -> Dict[str, Any]:
    service = get_translation_service()
    cases = load_cases(dataset_dir, "dataset_translation_multilingual.json", split)
    per_case = []
    line_match_sum = 0.0
    label_match_sum = 0.0

    for case in cases:
        _, segments = build_transcript(case)
        translated = service.translate_transcript(
            segments,
            source_lang=case.get("language", "auto"),
            target_lang=case.get("target_language", "en"),
        )
        pred_lines = format_translated_lines(translated)
        ref_lines = case["light_reference"]["reference_translation_style"]

        pair_count = min(len(pred_lines), len(ref_lines))
        structure_scores = []
        label_scores = []

        for idx in range(pair_count):
            pred_line = pred_lines[idx]
            ref_line = ref_lines[idx]
            structure_scores.append(SequenceMatcher(None, normalize_text(pred_line), normalize_text(ref_line)).ratio())

            pred_prefix = pred_line.split(":", 1)[0]
            ref_prefix = ref_line.split(":", 1)[0]
            label_scores.append(1.0 if normalize_text(pred_prefix) == normalize_text(ref_prefix) else 0.0)

        avg_line_score = sum(structure_scores) / len(structure_scores) if structure_scores else 0.0
        avg_label_score = sum(label_scores) / len(label_scores) if label_scores else 0.0
        line_match_sum += avg_line_score
        label_match_sum += avg_label_score

        per_case.append({
            "id": case["id"],
            "line_similarity": round(avg_line_score, 4),
            "speaker_timestamp_preservation": round(avg_label_score, 4),
            "predicted_lines": pred_lines,
        })

    total = len(cases)
    return {
        "num_cases": total,
        "summary": {
            "avg_line_similarity": round(line_match_sum / total, 4) if total else 0.0,
            "avg_speaker_timestamp_preservation": round(label_match_sum / total, 4) if total else 0.0,
        },
        "cases": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Smart Meeting benchmark.")
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Path to the benchmark dataset directory.",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "dev", "test"],
        help="Optional split filter.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["summarization", "action_items", "sentiment", "translation"],
        choices=["summarization", "action_items", "sentiment", "translation"],
        help="Tasks to evaluate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON report output path.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    results: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "tasks": args.tasks,
    }

    if "summarization" in args.tasks:
        results["summarization"] = evaluate_summarization(dataset_dir, args.split)
    if "action_items" in args.tasks:
        results["action_items"] = evaluate_action_items(dataset_dir, args.split)
    if "sentiment" in args.tasks:
        results["sentiment"] = evaluate_sentiment(dataset_dir, args.split)
    if "translation" in args.tasks:
        results["translation"] = evaluate_translation(dataset_dir, args.split)

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
