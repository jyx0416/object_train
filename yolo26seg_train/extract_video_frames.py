#!/usr/bin/env python3
"""Extract traceable, evenly sampled frames from the D405 videos.

The default rate is one image per second. Output names preserve both the source
video number and original frame number, which allows train/val/test to be split
by source video later without leaking adjacent frames between splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "video"
DEFAULT_OUTPUT_DIR = ROOT / "dataset" / "raw" / "images"
DEFAULT_MANIFEST = ROOT / "dataset" / "raw" / "frames.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Number of images to extract per video second (default: 1.0)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Only extract source videos not already recorded in the manifest",
    )
    return parser.parse_args()


def video_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.stem.removeprefix("video_")
    return (int(suffix), path.name) if suffix.isdigit() else (10**12, path.name)


def extract_video(video_path: Path, output_dir: Path, sample_fps: float) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"视频参数无效：{video_path}")

    # Sample at the center of each time interval instead of always taking the
    # first frame, which may contain camera auto-exposure transients.
    interval = source_fps / sample_fps
    target_positions: list[int] = []
    position = interval / 2
    while position < frame_count:
        frame_index = min(frame_count - 1, round(position))
        if not target_positions or frame_index != target_positions[-1]:
            target_positions.append(frame_index)
        position += interval
    if not target_positions:
        target_positions = [frame_count // 2]

    records: list[dict] = []
    for frame_index in target_positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"读取失败：{video_path.name} 第 {frame_index} 帧")

        image_name = f"{video_path.stem}_f{frame_index:06d}.png"
        image_path = output_dir / image_name
        if image_path.exists():
            capture.release()
            raise FileExistsError(f"拒绝覆盖已有图片：{image_path}")
        if not cv2.imwrite(str(image_path), frame):
            capture.release()
            raise RuntimeError(f"无法写入图片：{image_path}")

        records.append(
            {
                "image": image_name,
                "source_video": video_path.name,
                "frame_index": frame_index,
                "time_seconds": round(frame_index / source_fps, 6),
                "source_fps": source_fps,
                "width": width,
                "height": height,
            }
        )

    capture.release()
    return records


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps 必须大于 0")

    video_dir = args.videos.resolve()
    output_dir = args.output.resolve()
    manifest_path = args.manifest.resolve()
    videos = sorted(video_dir.glob("video_*.mp4"), key=video_sort_key)
    if not videos:
        raise SystemExit(f"没有找到视频：{video_dir}/video_*.mp4")

    existing = list(output_dir.glob("*.png")) if output_dir.exists() else []
    existing_manifest: dict | None = None
    if args.append:
        if not manifest_path.is_file():
            raise SystemExit(f"增量抽取需要已有来源清单：{manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_rate = float(existing_manifest.get("sample_fps", args.sample_fps))
        if abs(previous_rate - args.sample_fps) > 1e-9:
            raise SystemExit(
                f"抽帧率不一致：已有 {previous_rate} FPS，本次 {args.sample_fps} FPS"
            )
        processed_videos = {
            record["source_video"]
            for record in existing_manifest.get("records", [])
            if "source_video" in record
        }
        videos = [video for video in videos if video.name not in processed_videos]
        if not videos:
            print("没有尚未抽帧的新视频。")
            return
    elif existing:
        raise SystemExit(
            f"输出目录已有 {len(existing)} 张 PNG，为防止混入旧数据，本次未执行：{output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    new_records: list[dict] = []
    try:
        for index, video_path in enumerate(videos, start=1):
            video_records = extract_video(video_path, output_dir, args.sample_fps)
            new_records.extend(video_records)
            print(
                f"[{index:03d}/{len(videos):03d}] {video_path.name}: "
                f"{len(video_records)} 张"
            )
    except Exception:
        # Keep successfully extracted frames for diagnosis, but do not write a
        # misleading complete manifest.
        raise

    if existing_manifest is None:
        manifest = {
            "sample_fps": args.sample_fps,
            "images_dir": str(output_dir),
            "records": new_records,
        }
    else:
        manifest = existing_manifest
        manifest.setdefault("records", []).extend(new_records)

    manifest["video_count"] = len(
        {
            record["source_video"]
            for record in manifest["records"]
            if "source_video" in record
        }
    )
    manifest["image_count"] = len(list(output_dir.glob("*.png")))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"本次新增：{len(new_records)} 张图片")
    print(f"当前总数：{manifest['image_count']} 张图片")
    print(f"图片目录：{output_dir}")
    print(f"来源清单：{manifest_path}")


if __name__ == "__main__":
    main()
