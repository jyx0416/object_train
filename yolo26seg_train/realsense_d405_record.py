#!/usr/bin/env python3
"""Preview a RealSense D405 and record sequentially numbered color videos.

Keyboard controls (the preview window must have focus):
    1     Start recording a new video.
    2     Stop and save the current video.
    q/Esc Stop recording if needed and quit.

Videos are saved under ``yolo26seg_train/video`` as ``video_000000.mp4``,
``video_000001.mp4``, and so on. Existing recordings are never overwritten.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "video"
WIDTH = 640
HEIGHT = 480
FPS = 30
WINDOW_NAME = "RealSense D405 Video Recorder"


def next_video_path() -> Path:
    """Return the next sequence path without overwriting an existing video."""
    numbers: list[int] = []
    for path in VIDEO_DIR.glob("video_*.mp4"):
        match = re.fullmatch(r"video_(\d+)", path.stem)
        if match:
            numbers.append(int(match.group(1)))
    number = max(numbers, default=-1) + 1
    return VIDEO_DIR / f"video_{number:06d}.mp4"


def draw_status(
    frame: np.ndarray,
    recording: bool,
    output_path: Path | None,
    recorded_frames: int,
) -> np.ndarray:
    """Draw controls and recording state on the preview, not in saved video."""
    preview = frame.copy()
    if recording:
        state = f"REC  {recorded_frames / FPS:.1f}s   [2] stop and save"
        color = (0, 0, 255)
    else:
        state = "READY   [1] start recording"
        color = (0, 220, 0)

    cv2.rectangle(preview, (8, 8), (WIDTH - 8, 72), (0, 0, 0), -1)
    cv2.putText(
        preview,
        state,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    if output_path is not None:
        cv2.putText(
            preview,
            output_path.name,
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        preview,
        "1: start   2: stop/save   q/Esc: quit",
        (12, HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def create_writer(output_path: Path) -> cv2.VideoWriter:
    """Create and validate an MP4 writer for the configured color stream."""
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("无法创建 MP4 文件，请检查 OpenCV/FFmpeg 编码器。")
    return writer


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    writer: cv2.VideoWriter | None = None
    output_path: Path | None = None
    recorded_frames = 0
    camera_started = False

    try:
        profile = pipeline.start(config)
        camera_started = True
        device_name = profile.get_device().get_info(rs.camera_info.name)

        print(f"相机已启动：{device_name} ({WIDTH}x{HEIGHT} @ {FPS} FPS)")
        print(f"视频保存目录：{VIDEO_DIR}")
        print("按键：1=开始录制，2=结束并保存，q/Esc=退出")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Save the clean frame before adding the preview-only status text.
            if writer is not None:
                writer.write(frame)
                recorded_frames += 1

            preview = draw_status(
                frame,
                recording=writer is not None,
                output_path=output_path,
                recorded_frames=recorded_frames,
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("1"):
                if writer is None:
                    output_path = next_video_path()
                    writer = create_writer(output_path)
                    recorded_frames = 0
                    print(f"开始录制：{output_path.name}")
                else:
                    print("当前已经在录制；请先按 2 保存。")
            elif key == ord("2"):
                if writer is not None:
                    writer.release()
                    writer = None
                    duration = recorded_frames / FPS
                    print(
                        f"录制完成：{output_path.name}，"
                        f"{recorded_frames} 帧，约 {duration:.1f} 秒"
                    )
                    output_path = None
                    recorded_frames = 0
                else:
                    print("当前没有正在录制的视频；按 1 开始。")
            elif key in (ord("q"), 27):
                break

    except RuntimeError as exc:
        raise SystemExit(f"D405 采集失败：{exc}") from exc
    finally:
        if writer is not None:
            writer.release()
            duration = recorded_frames / FPS
            print(
                f"退出前已保存：{output_path.name}，"
                f"{recorded_frames} 帧，约 {duration:.1f} 秒"
            )
        if camera_started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
