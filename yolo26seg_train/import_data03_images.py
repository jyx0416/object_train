#!/usr/bin/env python3
"""Randomly import data03 simulation images until the raw pool reaches a target.

Sampling is stratified by environment and reproducible. Detection labels from
data03 are deliberately not copied because five-column box annotations are not
valid YOLO instance-segmentation labels.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_SOURCE = PROJECT_ROOT / "yolo_detect_train" / "data" / "data03" / "Images"
DEFAULT_OUTPUT = ROOT / "dataset" / "raw" / "images"
DEFAULT_MANIFEST = ROOT / "dataset" / "raw" / "frames.json"
DEFAULT_IMPORT_RECORD = ROOT / "dataset" / "raw" / "data03_import.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--target-total", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def environment(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def proportional_quotas(groups: dict[str, list[Path]], count: int) -> dict[str, int]:
    """Allocate an exact sample count proportionally using largest remainders."""
    total = sum(len(paths) for paths in groups.values())
    exact = {name: count * len(paths) / total for name, paths in groups.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(groups, key=lambda name: (exact[name] - quotas[name], name), reverse=True)
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    import_record_path = DEFAULT_IMPORT_RECORD.resolve()

    output.mkdir(parents=True, exist_ok=True)
    current_images = sorted(output.glob("*.png"))
    needed = args.target_total - len(current_images)
    if needed <= 0:
        print(f"当前已有 {len(current_images)} 张，无需补充（目标 {args.target_total} 张）。")
        return
    if import_record_path.exists():
        raise SystemExit(f"已存在导入记录，为避免重复抽样，本次未执行：{import_record_path}")

    source_images = sorted(source.glob("*.png"))
    existing_names = {path.name for path in current_images}
    candidates = [path for path in source_images if path.name not in existing_names]
    if len(candidates) < needed:
        raise SystemExit(f"可选图片只有 {len(candidates)} 张，无法补充 {needed} 张。")

    groups: dict[str, list[Path]] = defaultdict(list)
    for path in candidates:
        groups[environment(path)].append(path)
    quotas = proportional_quotas(groups, needed)

    rng = random.Random(args.seed)
    selected: list[Path] = []
    for name in sorted(groups):
        paths = groups[name]
        rng.shuffle(paths)
        selected.extend(paths[: quotas[name]])
    selected.sort()

    copied: list[dict] = []
    for source_path in selected:
        destination = output / source_path.name
        if destination.exists():
            raise FileExistsError(f"目标文件已存在：{destination}")
        shutil.copy2(source_path, destination)
        copied.append(
            {
                "image": destination.name,
                "source_dataset": "data03",
                "source_image": source_path.name,
                "environment": environment(source_path),
                "preferred_split": "train",
            }
        )

    import_record = {
        "source": str(source),
        "seed": args.seed,
        "target_total": args.target_total,
        "count": len(copied),
        "environment_counts": {
            name: sum(row["environment"] == name for row in copied) for name in sorted(groups)
        },
        "images": copied,
    }
    import_record_path.write_text(
        json.dumps(import_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("records", []).extend(copied)
        manifest["image_count"] = len(list(output.glob("*.png")))
        manifest["data03_import"] = {
            "count": len(copied),
            "seed": args.seed,
            "record": str(import_record_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"已从 data03 复制 {len(copied)} 张，总数达到 {len(current_images) + len(copied)} 张。")
    for name, count in import_record["environment_counts"].items():
        print(f"  {name}: {count} 张")
    print(f"抽样记录：{import_record_path}")


if __name__ == "__main__":
    main()
