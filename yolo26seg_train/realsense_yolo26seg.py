#!/usr/bin/env python3
"""RealSense D405 + YOLO26-seg real-time test. Press q/Esc to quit."""

from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


WEIGHTS = Path(__file__).parent / "runs/segment/final_n_v1/weights/best.pt"


def main() -> None:
    model = YOLO(WEIGHTS)
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline.start(config)
    try:
        while True:
            color = pipeline.wait_for_frames().get_color_frame()
            if not color:
                continue

            frame = np.asanyarray(color.get_data())
            result = model(frame, imgsz=640, conf=0.25, verbose=False)[0]
            cv2.imshow("D405 YOLO26-seg", result.plot())

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
