#!/usr/bin/env python3
"""Analyze EndoVis LocateAnything evaluation outputs.

The grounding evaluator writes an aggregate eval_results.json, while COCO-style
evaluation writes raw eval_results.jsonl predictions. This tool can summarize
the aggregate file alone, and can produce per-image failure diagnostics when
raw predictions are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]
DEFAULT_CLASS = "surgical instrument wrist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose low EndoVis LocateAnything mAP/localization performance."
    )
    parser.add_argument(
        "--metrics-json",
        "--eval-results",
        dest="metrics_json",
        default=None,
        help="Aggregate grounding eval_results.json. If omitted, only raw prediction diagnostics run.",
    )
    parser.add_argument(
        "--pred-jsonl",
        default=None,
        help="Raw prediction JSONL from inference, usually .../eval_results.jsonl.",
    )
    parser.add_argument(
        "--gt-jsonl",
        default=None,
        help="EndoVis val eval JSONL, usually annotations/endovis_val_eval.jsonl.",
    )
    parser.add_argument(
        "--coco-json",
        default=None,
        help="EndoVis COCO GT JSON, usually annotations/endovis_val_coco.json.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional image root used to draw worst-case overlays.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/endovis_eval",
        help="Directory for CSV/JSON/plot outputs.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_CLASS,
        help="Collapsed class label used in reports.",
    )
    parser.add_argument(
        "--category-aware",
        action="store_true",
        help="Match boxes only within the same category. Default is category-agnostic.",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=30,
        help="Number of worst samples to include in summary and optional overlays.",
    )
    parser.add_argument(
        "--duplicate-iou",
        type=float,
        default=0.90,
        help="IoU threshold for counting duplicate predictions.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots.",
    )
    parser.add_argument(
        "--draw-overlays",
        action="store_true",
        help="Draw GT/prediction overlays for the worst samples. Requires --image-root.",
    )
    return parser.parse_args()


def normalize_path(path: Any) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return mean(clean) if clean else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def normalize_box(raw: Any, xywh: bool = False) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    vals = [safe_float(v) for v in raw]
    if any(v is None for v in vals):
        return None
    x1, y1, x2, y2 = vals  # type: ignore[misc]
    if xywh:
        x2 = x1 + x2
        y2 = y1 + y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def read_json_or_jsonl(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL line: {exc}") from exc
        return rows


def is_aggregate_metrics(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and data
        and all(isinstance(k, str) for k in data)
        and any("basic_metrics" in v for v in data.values() if isinstance(v, dict))
    )


def summarize_metrics_json(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = read_json_or_jsonl(path)
    if not is_aggregate_metrics(data):
        raise ValueError(f"{path} does not look like an aggregate eval_results.json")

    rows: list[dict[str, Any]] = []
    dataset_keys = set()
    for threshold_text, payload in sorted(data.items(), key=lambda kv: float(kv[0])):
        basic = payload.get("basic_metrics", {})
        if not basic:
            continue
        dataset_key = next(iter(basic))
        dataset_keys.add(dataset_key)
        metrics = basic[dataset_key]
        precisions = [float(v) for v in metrics.get("precisions", [])]
        recalls = [float(v) for v in metrics.get("recalls", [])]
        gt_counts = [int(v) for v in metrics.get("gt_counts", [])]
        pred_counts = [int(v) for v in metrics.get("pred_counts", [])]
        precision = safe_mean(precisions)
        recall = safe_mean(recalls)
        row = {
            "iou": float(threshold_text),
            "dataset": dataset_key,
            "samples": len(recalls),
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "avg_gt_count": safe_mean([float(v) for v in gt_counts]),
            "avg_pred_count": safe_mean([float(v) for v in pred_counts]),
            "zero_recall_samples": sum(1 for v in recalls if v == 0),
            "perfect_recall_samples": sum(1 for v in recalls if v == 1),
            "zero_precision_samples": sum(1 for v in precisions if v == 0),
        }
        if payload.get("instruction_following_metrics", {}).get(dataset_key):
            ratios = payload["instruction_following_metrics"][dataset_key].get("ratios", [])
            row["instruction_following"] = safe_mean([float(v) for v in ratios])
        if payload.get("wrong_rejection_metrics", {}).get(dataset_key):
            wr = payload["wrong_rejection_metrics"][dataset_key].get("wrong_rejections", [])
            row["wrong_rejections"] = sum(int(v) for v in wr)
        rows.append(row)

    summary = {
        "source": str(path),
        "datasets": sorted(dataset_keys),
        "thresholds": [row["iou"] for row in rows],
        "macro_precision": safe_mean([row["precision"] for row in rows]),
        "macro_recall": safe_mean([row["recall"] for row in rows]),
        "macro_f1": safe_mean([row["f1"] for row in rows]),
        "samples": rows[0]["samples"] if rows else 0,
    }
    return rows, summary


def extract_box_dict(record: dict[str, Any], key: str) -> dict[str, list[list[float]]]:
    raw = record.get(key, {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, list[list[float]]] = {}
    for category, boxes in raw.items():
        if not isinstance(boxes, list):
            continue
        clean = []
        for box in boxes:
            normalized = normalize_box(box)
            if normalized is not None:
                clean.append(normalized)
        result[str(category)] = clean
    return result


def flatten_box_dict(boxes: dict[str, list[list[float]]], label: str) -> dict[str, list[list[float]]]:
    flat: list[list[float]] = []
    for category_boxes in boxes.values():
        flat.extend(category_boxes)
    return {label: flat}


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    data = read_json_or_jsonl(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict) and not is_aggregate_metrics(data):
        if "predictions" in data and isinstance(data["predictions"], list):
            return [row for row in data["predictions"] if isinstance(row, dict)]
        if "image_path" in data or "extracted_predictions" in data:
            return [data]
    raise ValueError(f"{path} does not look like raw prediction JSONL/JSON")


def load_gt_jsonl(path: str | Path) -> dict[str, dict[str, list[list[float]]]]:
    rows = read_json_or_jsonl(path)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not look like EndoVis GT JSONL")
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_path = row.get("image_path") or row.get("image")
        if not image_path:
            continue
        out[normalize_path(image_path)] = extract_box_dict(row, "gt")
    return out


def load_coco_gt(path: str | Path) -> dict[str, dict[str, list[list[float]]]]:
    data = read_json_or_jsonl(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not look like COCO JSON")
    categories = {cat["id"]: str(cat["name"]) for cat in data.get("categories", [])}
    image_id_to_path = {
        img["id"]: normalize_path(img["file_name"])
        for img in data.get("images", [])
        if "id" in img and "file_name" in img
    }
    out: dict[str, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    for ann in data.get("annotations", []):
        image_path = image_id_to_path.get(ann.get("image_id"))
        category = categories.get(ann.get("category_id"))
        box = normalize_box(ann.get("bbox"), xywh=True)
        if image_path and category and box:
            out[image_path][category].append(box)
    return {path: dict(boxes) for path, boxes in out.items()}


def add_basename_fallbacks(gt_by_path: dict[str, Any]) -> dict[str, Any]:
    out = dict(gt_by_path)
    basename_counts = Counter(os.path.basename(path) for path in gt_by_path)
    for path, value in gt_by_path.items():
        base = os.path.basename(path)
        if basename_counts[base] == 1:
            out.setdefault(base, value)
    return out


def load_gt_lookup(args: argparse.Namespace, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    gt_by_path: dict[str, Any] = {}
    if args.gt_jsonl:
        gt_by_path.update(load_gt_jsonl(args.gt_jsonl))
    if args.coco_json:
        gt_by_path.update(load_coco_gt(args.coco_json))
    if not gt_by_path:
        for row in predictions:
            image_path = row.get("image_path") or row.get("image")
            if image_path and isinstance(row.get("gt"), dict):
                gt_by_path[normalize_path(image_path)] = extract_box_dict(row, "gt")
    return add_basename_fallbacks(gt_by_path)


def resolve_gt_boxes(
    gt_lookup: dict[str, dict[str, list[list[float]]]],
    image_path: str,
    args: argparse.Namespace,
) -> dict[str, list[list[float]]] | None:
    candidates = [normalize_path(image_path), os.path.basename(image_path)]
    if args.image_root:
        image_root = normalize_path(args.image_root)
        normalized = normalize_path(image_path)
        if normalized.startswith(image_root.rstrip("/") + "/"):
            candidates.append(normalized[len(image_root.rstrip("/")) + 1 :])
        try:
            candidates.append(normalize_path(os.path.relpath(normalized, image_root)))
        except ValueError:
            pass

    for candidate in candidates:
        if candidate in gt_lookup:
            return gt_lookup[candidate]

    normalized = normalize_path(image_path)
    suffix_matches = [
        boxes
        for gt_path, boxes in gt_lookup.items()
        if normalized.endswith("/" + gt_path) or gt_path.endswith("/" + normalized)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def match_at_threshold(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    threshold: float,
) -> tuple[int, set[int], set[int]]:
    pairs = []
    for gt_idx, gt in enumerate(gt_boxes):
        for pred_idx, pred in enumerate(pred_boxes):
            pairs.append((box_iou(gt, pred), gt_idx, pred_idx))
    pairs.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for iou, gt_idx, pred_idx in pairs:
        if iou < threshold:
            break
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
    return len(matched_gt), matched_gt, matched_pred


def best_pairs(gt_boxes: list[list[float]], pred_boxes: list[list[float]]) -> list[dict[str, Any]]:
    rows = []
    for gt_idx, gt in enumerate(gt_boxes):
        best_iou = 0.0
        best_pred_idx = None
        best_pred = None
        for pred_idx, pred in enumerate(pred_boxes):
            iou = box_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx
                best_pred = pred
        rows.append(
            {
                "gt_idx": gt_idx,
                "gt_box": gt,
                "pred_idx": best_pred_idx,
                "pred_box": best_pred,
                "best_iou": best_iou,
            }
        )
    return rows


def duplicate_count(pred_boxes: list[list[float]], threshold: float) -> int:
    count = 0
    for i, left in enumerate(pred_boxes):
        for right in pred_boxes[i + 1 :]:
            if box_iou(left, right) >= threshold:
                count += 1
    return count


def box_size_bucket(box: list[float]) -> str:
    area = box_area(box)
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def frame_index(image_path: str) -> int | None:
    try:
        return int(Path(image_path).stem)
    except ValueError:
        return None


def video_name(image_path: str) -> str:
    parent = Path(image_path).parent.as_posix()
    return parent if parent and parent != "." else "unknown"


def analyze_raw_predictions(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = load_prediction_records(args.pred_jsonl)
    gt_lookup = load_gt_lookup(args, predictions)
    if not gt_lookup:
        raise ValueError("No GT found. Pass --gt-jsonl/--coco-json or use predictions with embedded gt.")

    sample_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    threshold_totals = {
        thr: {"matches": 0, "gt": 0, "pred": 0}
        for thr in IOU_THRESHOLDS
    }
    unmatched_predictions = 0
    missing_gt_samples = 0

    for sample_idx, row in enumerate(predictions):
        image_path = normalize_path(row.get("image_path") or row.get("image") or row.get("image_name") or sample_idx)
        gt_boxes_by_cat = resolve_gt_boxes(gt_lookup, image_path, args)
        if gt_boxes_by_cat is None:
            gt_boxes_by_cat = extract_box_dict(row, "gt")
            if not gt_boxes_by_cat:
                missing_gt_samples += 1
        pred_boxes_by_cat = extract_box_dict(row, "extracted_predictions")
        if not args.category_aware:
            gt_boxes_by_cat = flatten_box_dict(gt_boxes_by_cat, args.label)
            pred_boxes_by_cat = flatten_box_dict(pred_boxes_by_cat, args.label)

        categories = sorted(set(gt_boxes_by_cat) | set(pred_boxes_by_cat))
        sample_gt_count = sum(len(gt_boxes_by_cat.get(cat, [])) for cat in categories)
        sample_pred_count = sum(len(pred_boxes_by_cat.get(cat, [])) for cat in categories)
        sample_matches_50 = 0
        sample_matches_75 = 0
        sample_best_ious: list[float] = []
        sample_duplicates = 0
        failure_modes: set[str] = set()

        for category in categories:
            gt_boxes = gt_boxes_by_cat.get(category, [])
            pred_boxes = pred_boxes_by_cat.get(category, [])
            sample_duplicates += duplicate_count(pred_boxes, args.duplicate_iou)
            for thr in IOU_THRESHOLDS:
                matches, _, _ = match_at_threshold(gt_boxes, pred_boxes, thr)
                threshold_totals[thr]["matches"] += matches
                threshold_totals[thr]["gt"] += len(gt_boxes)
                threshold_totals[thr]["pred"] += len(pred_boxes)
                if thr == 0.50:
                    sample_matches_50 += matches
                elif thr == 0.75:
                    sample_matches_75 += matches

            for pair in best_pairs(gt_boxes, pred_boxes):
                gt_box = pair["gt_box"]
                pred_box = pair["pred_box"]
                best_iou = float(pair["best_iou"])
                sample_best_ious.append(best_iou)
                matched_50 = best_iou >= 0.50
                matched_75 = best_iou >= 0.75
                row_out = {
                    "image_path": image_path,
                    "video": video_name(image_path),
                    "frame_index": frame_index(image_path),
                    "category": category,
                    "gt_idx": pair["gt_idx"],
                    "pred_idx": pair["pred_idx"],
                    "best_iou": best_iou,
                    "matched_50": int(matched_50),
                    "matched_75": int(matched_75),
                    "gt_area": box_area(gt_box),
                    "gt_size": box_size_bucket(gt_box),
                    "gt_x1": gt_box[0],
                    "gt_y1": gt_box[1],
                    "gt_x2": gt_box[2],
                    "gt_y2": gt_box[3],
                }
                if pred_box:
                    gt_cx = (gt_box[0] + gt_box[2]) / 2
                    gt_cy = (gt_box[1] + gt_box[3]) / 2
                    pred_cx = (pred_box[0] + pred_box[2]) / 2
                    pred_cy = (pred_box[1] + pred_box[3]) / 2
                    gt_w = gt_box[2] - gt_box[0]
                    gt_h = gt_box[3] - gt_box[1]
                    pred_w = pred_box[2] - pred_box[0]
                    pred_h = pred_box[3] - pred_box[1]
                    row_out.update(
                        {
                            "pred_x1": pred_box[0],
                            "pred_y1": pred_box[1],
                            "pred_x2": pred_box[2],
                            "pred_y2": pred_box[3],
                            "center_dx": pred_cx - gt_cx,
                            "center_dy": pred_cy - gt_cy,
                            "width_ratio": pred_w / gt_w if gt_w else 0.0,
                            "height_ratio": pred_h / gt_h if gt_h else 0.0,
                            "area_ratio": box_area(pred_box) / box_area(gt_box) if box_area(gt_box) else 0.0,
                            "left_error": pred_box[0] - gt_box[0],
                            "top_error": pred_box[1] - gt_box[1],
                            "right_error": pred_box[2] - gt_box[2],
                            "bottom_error": pred_box[3] - gt_box[3],
                        }
                    )
                box_rows.append(row_out)

        if sample_pred_count == 0 and sample_gt_count > 0:
            failure_modes.add("empty_prediction")
        if sample_matches_50 < sample_gt_count:
            failure_modes.add("missed_gt")
        if sample_matches_50 < sample_pred_count:
            failure_modes.add("false_positive")
        if sample_pred_count < sample_gt_count:
            failure_modes.add("count_under")
        if sample_pred_count > sample_gt_count:
            failure_modes.add("count_over")
        if sample_duplicates:
            failure_modes.add("duplicate_prediction")
        if sample_gt_count and sample_matches_50 > sample_matches_75:
            failure_modes.add("good_at_50_bad_at_75")
        if sample_gt_count and sample_best_ious and max(sample_best_ious) < 0.50:
            failure_modes.add("low_iou_localization")
        if not failure_modes:
            failure_modes.add("ok")

        unmatched_predictions += max(0, sample_pred_count - sample_matches_50)
        best_iou_mean = safe_mean(sample_best_ious)
        sample_rows.append(
            {
                "sample_idx": sample_idx,
                "image_path": image_path,
                "video": video_name(image_path),
                "frame_index": frame_index(image_path),
                "gt_count": sample_gt_count,
                "pred_count": sample_pred_count,
                "matches_50": sample_matches_50,
                "matches_75": sample_matches_75,
                "precision_50": sample_matches_50 / sample_pred_count if sample_pred_count else 0.0,
                "recall_50": sample_matches_50 / sample_gt_count if sample_gt_count else 0.0,
                "best_iou_mean": best_iou_mean,
                "best_iou_max": max(sample_best_ious) if sample_best_ious else 0.0,
                "duplicate_pairs": sample_duplicates,
                "failure_modes": ";".join(sorted(failure_modes)),
            }
        )

    threshold_rows = []
    for thr in IOU_THRESHOLDS:
        totals = threshold_totals[thr]
        precision = totals["matches"] / totals["pred"] if totals["pred"] else 0.0
        recall = totals["matches"] / totals["gt"] if totals["gt"] else 0.0
        threshold_rows.append(
            {
                "iou": thr,
                "matches": totals["matches"],
                "gt": totals["gt"],
                "pred": totals["pred"],
                "precision": precision,
                "recall": recall,
                "f1": f1_score(precision, recall),
            }
        )

    best_ious = [row["best_iou"] for row in box_rows]
    failure_counter = Counter()
    for row in sample_rows:
        failure_counter.update(row["failure_modes"].split(";"))

    summary = {
        "source": args.pred_jsonl,
        "samples": len(sample_rows),
        "gt_boxes": sum(row["gt_count"] for row in sample_rows),
        "pred_boxes": sum(row["pred_count"] for row in sample_rows),
        "unmatched_predictions_at_50": unmatched_predictions,
        "missing_gt_samples": missing_gt_samples,
        "best_iou_mean": safe_mean(best_ious),
        "best_iou_median": median(best_ious) if best_ious else 0.0,
        "best_iou_p10": percentile(best_ious, 10),
        "best_iou_p90": percentile(best_ious, 90),
        "failure_modes": dict(failure_counter),
        "category_aware": bool(args.category_aware),
    }
    return summary, threshold_rows, sample_rows, box_rows


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return 0.0
    index = (len(clean) - 1) * pct / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return clean[int(index)]
    return clean[lo] * (hi - index) + clean[hi] * (index - lo)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"Skipping plots: could not import matplotlib ({exc})")
        return None


def plot_thresholds(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    plt = maybe_import_matplotlib()
    if plt is None or not rows:
        return
    x = [row["iou"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, color in [("precision", "#c43b3b"), ("recall", "#2e7d32"), ("f1", "#2f5f9f")]:
        ax.plot(x, [row[metric] for row in rows], marker="o", label=metric, color=color)
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_histogram(path: Path, values: list[float], title: str, xlabel: str) -> None:
    plt = maybe_import_matplotlib()
    if plt is None or not values:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=20, color="#557a95", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_center_offsets(path: Path, box_rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    xs = [row.get("center_dx") for row in box_rows if row.get("center_dx") is not None]
    ys = [row.get("center_dy") for row in box_rows if row.get("center_dy") is not None]
    if plt is None or not xs or not ys:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys, s=12, alpha=0.55, color="#5b6fba")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Prediction center offset from GT")
    ax.set_xlabel("dx pixels")
    ax.set_ylabel("dy pixels")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_video(path: Path, sample_rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    if plt is None or not sample_rows:
        return
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in sample_rows:
        grouped[row["video"]].append(float(row["best_iou_mean"]))
    videos = sorted(grouped)
    values = [safe_mean(grouped[v]) for v in videos]
    fig_width = max(7, min(18, len(videos) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.bar(videos, values, color="#6c8f4e")
    ax.set_ylim(0, 1)
    ax.set_title("Mean best IoU by video")
    ax.set_ylabel("Mean best IoU")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scale_bias(path: Path, box_rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    metrics = ["width_ratio", "height_ratio", "area_ratio"]
    values = [[row[m] for row in box_rows if row.get(m) is not None] for m in metrics]
    if not any(values):
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, metric, vals in zip(axes, metrics, values):
        ax.hist(vals, bins=20, color="#b7834c", edgecolor="white")
        ax.axvline(1.0, color="black", linewidth=0.8)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_count_errors(path: Path, sample_rows: list[dict[str, Any]]) -> None:
    plt = maybe_import_matplotlib()
    values = [row["pred_count"] - row["gt_count"] for row in sample_rows]
    if plt is None or not values:
        return
    bins = range(min(values) - 1, max(values) + 2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, color="#8f5e99", edgecolor="white", align="left")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Prediction count error")
    ax.set_xlabel("pred_count - gt_count")
    ax.set_ylabel("Samples")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_overlays(
    out_dir: Path,
    sample_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not args.image_root or not args.draw_overlays:
        return
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"Skipping overlays: could not import PIL ({exc})")
        return

    pred_by_image = defaultdict(list)
    for row in prediction_rows:
        pred_by_image[row["image_path"]].append(row)

    overlay_dir = out_dir / "failure_montage"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted(
        sample_rows,
        key=lambda row: (row["recall_50"], row["precision_50"], row["best_iou_mean"]),
    )[: args.worst_n]

    for rank, sample in enumerate(worst, start=1):
        image_path = sample["image_path"]
        source = Path(args.image_root) / image_path
        if not source.exists():
            source = Path(args.image_root) / os.path.basename(image_path)
        if not source.exists():
            continue
        try:
            image = Image.open(source).convert("RGB")
        except Exception:
            continue
        draw = ImageDraw.Draw(image)
        for row in pred_by_image[image_path]:
            gt = [row["gt_x1"], row["gt_y1"], row["gt_x2"], row["gt_y2"]]
            draw.rectangle(gt, outline=(0, 210, 80), width=3)
            if row.get("pred_x1") is not None:
                pred = [row["pred_x1"], row["pred_y1"], row["pred_x2"], row["pred_y2"]]
                draw.rectangle(pred, outline=(230, 50, 50), width=3)
                draw.text((pred[0], max(0, pred[1] - 14)), f"IoU {row['best_iou']:.2f}", fill=(230, 50, 50))
        safe_name = normalize_path(image_path).replace("/", "__")
        image.save(overlay_dir / f"{rank:03d}_{safe_name}")


def write_summary_markdown(
    path: Path,
    aggregate_summary: dict[str, Any] | None,
    raw_summary: dict[str, Any] | None,
    aggregate_rows: list[dict[str, Any]],
    raw_threshold_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> None:
    lines = ["# EndoVis Evaluation Analysis", ""]
    if aggregate_summary:
        lines.extend(
            [
                "## Aggregate Grounding Metrics",
                "",
                f"- Source: `{aggregate_summary['source']}`",
                f"- Samples: {aggregate_summary['samples']}",
                f"- Mean precision over thresholds: {aggregate_summary['macro_precision']:.4f}",
                f"- Mean recall over thresholds: {aggregate_summary['macro_recall']:.4f}",
                f"- Mean F1 over thresholds: {aggregate_summary['macro_f1']:.4f}",
                "",
                "| IoU | Precision | Recall | F1 | Zero Recall Samples |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in aggregate_rows:
            lines.append(
                f"| {row['iou']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['zero_recall_samples']} |"
            )
        lines.append("")

    if raw_summary:
        lines.extend(
            [
                "## Raw Prediction Diagnostics",
                "",
                f"- Source: `{raw_summary['source']}`",
                f"- Samples: {raw_summary['samples']}",
                f"- GT boxes: {raw_summary['gt_boxes']}",
                f"- Predicted boxes: {raw_summary['pred_boxes']}",
                f"- Mean best IoU: {raw_summary['best_iou_mean']:.4f}",
                f"- Median best IoU: {raw_summary['best_iou_median']:.4f}",
                f"- Best IoU p10/p90: {raw_summary['best_iou_p10']:.4f} / {raw_summary['best_iou_p90']:.4f}",
                "",
                "| IoU | Precision | Recall | F1 | Matches |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in raw_threshold_rows:
            lines.append(
                f"| {row['iou']:.2f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['matches']} |"
            )
        lines.extend(["", "### Failure Modes", ""])
        for mode, count in sorted(raw_summary["failure_modes"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {mode}: {count}")
        lines.extend(["", "### Worst Samples", ""])
        lines.append("| Image | GT | Pred | P@0.50 | R@0.50 | Best IoU | Modes |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        worst = sorted(
            sample_rows,
            key=lambda row: (row["recall_50"], row["precision_50"], row["best_iou_mean"]),
        )[:20]
        for row in worst:
            lines.append(
                f"| `{row['image_path']}` | {row['gt_count']} | {row['pred_count']} | "
                f"{row['precision_50']:.3f} | {row['recall_50']:.3f} | "
                f"{row['best_iou_mean']:.3f} | {row['failure_modes']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_summary: dict[str, Any] | None = None
    if args.metrics_json:
        aggregate_rows, aggregate_summary = summarize_metrics_json(args.metrics_json)
        write_csv(out_dir / "threshold_metrics.csv", aggregate_rows)
        write_json(out_dir / "threshold_metrics.json", aggregate_rows)
        if not args.no_plots:
            plot_thresholds(out_dir / "precision_recall_f1_over_iou.png", aggregate_rows, "Aggregate grounding metrics")

    raw_summary: dict[str, Any] | None = None
    raw_threshold_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    if args.pred_jsonl:
        raw_summary, raw_threshold_rows, sample_rows, box_rows = analyze_raw_predictions(args)
        write_csv(out_dir / "raw_threshold_metrics.csv", raw_threshold_rows)
        write_csv(out_dir / "sample_errors.csv", sample_rows)
        write_csv(out_dir / "box_diagnostics.csv", box_rows)
        write_json(out_dir / "raw_summary.json", raw_summary)
        if not args.no_plots:
            plot_thresholds(out_dir / "raw_precision_recall_f1_over_iou.png", raw_threshold_rows, "Raw prediction metrics")
            plot_histogram(
                out_dir / "best_iou_histogram.png",
                [row["best_iou"] for row in box_rows],
                "Best IoU per GT box",
                "Best IoU",
            )
            plot_center_offsets(out_dir / "center_offset_scatter.png", box_rows)
            plot_scale_bias(out_dir / "scale_bias_histograms.png", box_rows)
            plot_per_video(out_dir / "per_video_metrics.png", sample_rows)
            plot_count_errors(out_dir / "count_error_histogram.png", sample_rows)
        draw_overlays(out_dir, sample_rows, box_rows, args)

    combined_summary = {
        "aggregate": aggregate_summary,
        "raw": raw_summary,
        "outputs": {
            "summary_md": str(out_dir / "summary.md"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    write_json(out_dir / "summary.json", combined_summary)
    write_summary_markdown(
        out_dir / "summary.md",
        aggregate_summary,
        raw_summary,
        aggregate_rows,
        raw_threshold_rows,
        sample_rows,
    )

    print(f"Wrote EndoVis eval analysis to: {out_dir}")
    if aggregate_summary:
        print(
            "Aggregate mean over IoU thresholds: "
            f"P={aggregate_summary['macro_precision']:.4f}, "
            f"R={aggregate_summary['macro_recall']:.4f}, "
            f"F1={aggregate_summary['macro_f1']:.4f}"
        )
    if raw_summary:
        print(
            "Raw prediction summary: "
            f"samples={raw_summary['samples']}, "
            f"gt={raw_summary['gt_boxes']}, pred={raw_summary['pred_boxes']}, "
            f"mean_best_iou={raw_summary['best_iou_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
