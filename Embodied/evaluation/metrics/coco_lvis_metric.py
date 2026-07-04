# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import json
import os

import fastevaluate as fe
import numpy as np


def safe_mean(values):
    vals = [v for v in values if v >= 0 and np.isfinite(v)]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


def format_table(rows, headers):
    # Compute column widths
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]

    def fmt_row(row):
        return " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


def normalize_category_name(name):
    return str(name).lower().replace("_", " ")


def load_tsv_as_coco_detections(gt_path, pred_tsv_path):
    from pycocotools.coco import COCO

    coco_gt = COCO(gt_path)
    name_to_id = {
        normalize_category_name(cat["name"]): cat_id
        for cat_id, cat in coco_gt.cats.items()
    }

    detections = []
    with open(pred_tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                image_id_str, payload = line.split("\t", 1)
                image_id = int(image_id_str)
                items = json.loads(payload)
            except Exception:
                continue

            for item in items:
                category_name = normalize_category_name(item.get("class", ""))
                if category_name not in name_to_id:
                    continue
                bbox = item.get("rect")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                x, y, w, h = [float(v) for v in bbox]
                if w <= 0 or h <= 0:
                    continue
                detections.append(
                    {
                        "image_id": image_id,
                        "category_id": name_to_id[category_name],
                        "bbox": [x, y, w, h],
                        "score": float(item.get("conf", 1.0)),
                    }
                )

    return coco_gt, detections


def calculate_cocoeval_per_iou(gt_path, pred_tsv_path):
    from pycocotools.cocoeval import COCOeval

    coco_gt, detections = load_tsv_as_coco_detections(gt_path, pred_tsv_path)
    if not detections:
        raise ValueError(f"No valid detections found in {pred_tsv_path}")

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    area_idx = list(evaluator.params.areaRngLbl).index("all")
    max_det_idx = len(evaluator.params.maxDets) - 1

    rows = []
    for iou_idx, iou in enumerate(evaluator.params.iouThrs):
        p = precision[iou_idx, :, :, area_idx, max_det_idx]
        p = p[p > -1]
        ap = float(np.mean(p)) if p.size else 0.0

        r = recall[iou_idx, :, area_idx, max_det_idx]
        r = r[r > -1]
        ar = float(np.mean(r)) if r.size else 0.0

        rows.append(
            {
                "iou": float(iou),
                "ap": ap,
                "ar": ar,
                "f1": f1(ap, ar),
            }
        )
    return rows


def save_per_iou_json(rows, output_path):
    if not output_path:
        return
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved per-IoU metrics JSON to: {output_path}")


def save_per_iou_plot(rows, output_path):
    if not output_path:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping per-IoU plot because matplotlib is unavailable: {exc}")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ious = [row["iou"] for row in rows]
    aps = [row["ap"] for row in rows]
    ars = [row["ar"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ious, aps, marker="o", label="AP")
    ax.plot(ious, ars, marker="s", label="AR")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("Score")
    ax.set_title("COCOeval AP/AR over IoU thresholds")
    ax.set_xticks(ious)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved per-IoU plot to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate COCO/LVIS predictions (FastEval)."
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="path/to/EvalData/coco/instances_val2017.json",
        help="Path to GT json (COCO/LVIS)",
    )
    parser.add_argument(
        "--pred_tsv",
        type=str,
        default="path/to/EvalData/eval_results/box_eval/COCO/fast_eval.tsv",
        help="Path to predictions (TSV/JSON)",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="auto",
        choices=["auto", "coco", "lvis"],
        help="Evaluation type: auto (detect from filename), coco, or lvis",
    )
    parser.add_argument(
        "--per_iou_json",
        type=str,
        default=None,
        help="Optional path to save AP/AR/F1 at each COCO IoU threshold as JSON.",
    )
    parser.add_argument(
        "--per_iou_plot",
        type=str,
        default=None,
        help="Optional path to save AP/AR-over-IoU plot as PNG.",
    )
    parser.add_argument(
        "--skip_per_iou",
        action="store_true",
        help="Disable pycocotools per-IoU table/JSON/plot generation.",
    )
    return parser.parse_args()


def detect_eval_type(gt_path, pred_path, eval_type):
    """Detect evaluation type from file paths if auto is specified"""
    if eval_type != "auto":
        return eval_type

    # Check for LVIS indicators in paths
    lvis_indicators = ["lvis", "LVIS"]
    for indicator in lvis_indicators:
        if indicator in gt_path or indicator in pred_path:
            return "lvis"

    # Default to COCO
    return "coco"


def main():
    args = parse_args()

    # Detect evaluation type
    eval_type = detect_eval_type(args.gt, args.pred_tsv, args.eval_type)
    print(f"🔍 Detected evaluation type: {eval_type.upper()}")

    res = fe.evaluate(args.gt, args.pred_tsv, 0, 0, eval_type)

    per_iou_rows = None
    if args.skip_per_iou:
        print("\nSkipping per-IoU COCOeval metrics because --skip_per_iou was set.")
    else:
        try:
            per_iou_rows = calculate_cocoeval_per_iou(args.gt, args.pred_tsv)
        except Exception as exc:
            print(f"\nSkipping per-IoU COCOeval metrics: {exc}")
            print("Install pycocotools to enable this table/plot if it is missing.")

    if per_iou_rows:
        map_value = float(np.mean([row["ap"] for row in per_iou_rows]))
        mar_value = float(np.mean([row["ar"] for row in per_iou_rows]))
        ap50 = next(row["ap"] for row in per_iou_rows if round(row["iou"], 2) == 0.50)
        ar50 = next(row["ar"] for row in per_iou_rows if round(row["iou"], 2) == 0.50)
        ap95 = next(row["ap"] for row in per_iou_rows if round(row["iou"], 2) == 0.95)
        ar95 = next(row["ar"] for row in per_iou_rows if round(row["iou"], 2) == 0.95)
        summary_source = "pycocotools COCOeval"
    else:
        map_value = safe_mean(res.get("ap", []))
        mar_value = safe_mean(res.get("recall", []))
        ap50 = safe_mean(res.get("ap50", []))
        ar50 = safe_mean(res.get("recall50", []))
        ap95 = None
        ar95 = safe_mean(res.get("recall95", []))
        summary_source = "FastEvaluate fallback"

    headers = ["Metric", "Value"]
    rows = [
        ["Source", summary_source],
        ["mAP@[.50:.95]", f"{map_value:.4f}"],
        ["mAR@[.50:.95]", f"{mar_value:.4f}"],
        ["F1(mAP,mAR)", f"{f1(map_value, mar_value):.4f}"],
        ["AP@0.50", f"{ap50:.4f}"],
        ["AR@0.50", f"{ar50:.4f}"],
        ["F1@0.50", f"{f1(ap50, ar50):.4f}"],
        ["AP@0.95", "n/a" if ap95 is None else f"{ap95:.4f}"],
        ["AR@0.95", f"{ar95:.4f}"],
        ["F1@0.95", "n/a" if ap95 is None else f"{f1(ap95, ar95):.4f}"],
    ]

    print(format_table(rows, headers))

    legacy_rows = [
        ["FastEval ap", f"{safe_mean(res.get('ap', [])):.4f}"],
        ["FastEval ap50", f"{safe_mean(res.get('ap50', [])):.4f}"],
        ["FastEval endpoint precision50", f"{safe_mean(res.get('precision50', [])):.4f}"],
        ["FastEval endpoint recall50", f"{safe_mean(res.get('recall50', [])):.4f}"],
    ]
    print("\nFastEvaluate raw summary fields")
    print(format_table(legacy_rows, ["Field", "Value"]))

    if not per_iou_rows:
        return

    per_iou_table = [
        [
            f"{row['iou']:.2f}",
            f"{row['ap']:.4f}",
            f"{row['ar']:.4f}",
            f"{row['f1']:.4f}",
        ]
        for row in per_iou_rows
    ]
    print("\nPer-IoU COCOeval metrics")
    print(format_table(per_iou_table, ["IoU", "AP", "AR", "F1"]))

    save_per_iou_json(per_iou_rows, args.per_iou_json)
    save_per_iou_plot(per_iou_rows, args.per_iou_plot)


if __name__ == "__main__":
    main()
