#!/usr/bin/env python3
"""Prepare EndoVis COCO annotations for LocateAnything fine-tuning.

The LocateAnything trainer consumes ShareGPT-style JSONL samples with
normalized coordinate tokens. This script converts the local EndoVis
COCO files and optionally extracts the referenced MP4 frames.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endovis-dir", default="data/endovis")
    parser.add_argument("--output-dir", default="data/endovis_locany")
    parser.add_argument(
        "--val-videos",
        default="7_fps1",
        help="Comma-separated video stems to hold out from train JSONL.",
    )
    parser.add_argument(
        "--modes",
        default="detection,grounding",
        help="Comma-separated sample types: detection, grounding.",
    )
    parser.add_argument(
        "--collapse-class",
        action="store_true",
        help='Rename every EndoVis category to the collapsed single class label.',
    )
    parser.add_argument(
        "--collapsed-label",
        default="surgical instrument wrist",
        help="Class label used when --collapse-class is set.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth annotated frame per video. The first annotated frame is always kept.",
    )
    parser.add_argument(
        "--skip-frame-extraction",
        action="store_true",
        help="Only write JSONL/recipe files. Use if frames already exist.",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Rewrite frame JPEGs even when they already exist.",
    )
    return parser.parse_args()


def norm_coord(value: float, size: int) -> int:
    return max(0, min(1000, int(round(value / size * 1000))))


def convert_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1 = norm_coord(x, width)
    y1 = norm_coord(y, height)
    x2 = norm_coord(x + w, width)
    y2 = norm_coord(y + h, height)
    x2 = max(x1 + 1, min(1000, x2))
    y2 = max(y1 + 1, min(1000, y2))
    return x1, y1, x2, y2


def box_text(box: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = box
    return f"<box><{x1}><{y1}><{x2}><{y2}></box>"


def response_for_instances(instances: list[tuple[str, tuple[int, int, int, int]]]) -> str:
    return "".join(f"<ref>{name}</ref>{box_text(box)}" for name, box in instances)


def extract_frames(video_path: Path, output_dir: Path, frame_names: set[str], overwrite: bool) -> None:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    needed = {
        int(Path(name).stem): output_dir / name
        for name in frame_names
        if overwrite or not (output_dir / name).exists()
    }
    if not needed:
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    needed_indices = sorted(needed)
    needed_set = set(needed_indices)
    max_needed = needed_indices[-1]

    frame_idx = 0
    while frame_idx <= max_needed:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        if frame_idx in needed_set:
            if not cv2.imwrite(str(needed[frame_idx]), frame):
                raise RuntimeError(f"Could not write frame: {needed[frame_idx]}")
        frame_idx += 1

    cap.release()


def make_samples(
    coco_path: Path,
    modes: set[str],
    collapse_class: bool,
    collapsed_label: str,
    frame_stride: int,
) -> tuple[list[dict], set[str]]:
    data = json.loads(coco_path.read_text())
    video_stem = coco_path.name.removesuffix("_coco.json")
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    images = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)

    samples = []
    frame_names = set()
    image_ids = sorted(
        anns_by_image,
        key=lambda image_id: int(Path(images[image_id]["file_name"]).stem),
    )
    for ordinal, image_id in enumerate(image_ids):
        if ordinal % frame_stride != 0:
            continue

        anns = anns_by_image[image_id]
        image = images[image_id]
        width, height = int(image["width"]), int(image["height"])
        rel_image = f"{video_stem}/{image['file_name']}"
        frame_names.add(image["file_name"])

        instances = []
        by_category: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
        for ann in anns:
            name = collapsed_label if collapse_class else categories[ann["category_id"]]
            box = convert_box(ann["bbox"], width, height)
            instances.append((name, box))
            by_category[name].append(box)

        if "detection" in modes:
            category_prompt = "</c>".join(sorted(by_category))
            samples.append(
                {
                    "image": rel_image,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "Locate all the instances that matches the following "
                                f"description: {category_prompt}."
                            ),
                        },
                        {"from": "gpt", "value": response_for_instances(instances)},
                    ],
                }
            )

        if "grounding" in modes:
            for category, boxes in sorted(by_category.items()):
                cat_instances = [(category, box) for box in boxes]
                samples.append(
                    {
                        "image": rel_image,
                        "conversations": [
                            {
                                "from": "human",
                                "value": (
                                    "Locate all the instances that match the following "
                                    f"description: {category}."
                                ),
                            },
                            {"from": "gpt", "value": response_for_instances(cat_instances)},
                        ],
                    }
                )

    return samples, frame_names


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    endovis_dir = Path(args.endovis_dir)
    output_dir = Path(args.output_dir)
    image_root = output_dir / "images"
    ann_dir = output_dir / "annotations"
    modes = {mode.strip() for mode in args.modes.split(",") if mode.strip()}
    val_videos = {stem.strip() for stem in args.val_videos.split(",") if stem.strip()}

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    invalid_modes = modes - {"detection", "grounding"}
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {sorted(invalid_modes)}")

    train_samples: list[dict] = []
    val_samples: list[dict] = []
    total_frames = 0

    coco_paths = sorted(glob.glob(str(endovis_dir / "*_fps1_coco.json")))
    if not coco_paths:
        raise FileNotFoundError(f"No *_fps1_coco.json files found in {endovis_dir}")

    for coco_str in coco_paths:
        coco_path = Path(coco_str)
        video_stem = coco_path.name.removesuffix("_coco.json")
        samples, frame_names = make_samples(
            coco_path,
            modes,
            collapse_class=args.collapse_class,
            collapsed_label=args.collapsed_label,
            frame_stride=args.frame_stride,
        )
        if video_stem in val_videos:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

        if not args.skip_frame_extraction:
            video_path = endovis_dir / f"{video_stem}.mp4"
            extract_frames(
                video_path,
                image_root / video_stem,
                frame_names,
                overwrite=args.overwrite_frames,
            )
        total_frames += len(frame_names)

    train_jsonl = ann_dir / "endovis_train.jsonl"
    val_jsonl = ann_dir / "endovis_val.jsonl"
    recipe_path = output_dir / "endovis_recipe.json"
    write_jsonl(train_jsonl, train_samples)
    write_jsonl(val_jsonl, val_samples)

    recipe = {
        "endovis_instruments": {
            "annotation": str(train_jsonl.resolve()),
            "root": str(image_root.resolve()),
            "repeat_time": 1.0,
            "data_augment": True,
        }
    }
    recipe_path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(train_samples)} train samples: {train_jsonl}")
    print(f"Wrote {len(val_samples)} val samples: {val_jsonl}")
    print(f"Wrote recipe: {recipe_path}")
    print(f"Referenced {total_frames} frames under: {image_root}")
    print(f"Frame stride: {args.frame_stride}")
    if args.collapse_class:
        print(f"Collapsed all categories to: {args.collapsed_label}")


if __name__ == "__main__":
    main()
