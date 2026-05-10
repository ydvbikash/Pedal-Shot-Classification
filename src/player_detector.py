from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np

import config


@dataclass
class Player:
    player_id: str
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2
    center: tuple[int, int]


class PlayerDetector:
    """Detects players using YOLOv8 when available; otherwise uses a simple color/motion fallback.

    The fallback keeps the project runnable even on machines where the YOLO model cannot be downloaded.
    For final submission, install requirements and use YOLOv8 for better player detection.
    """

    def __init__(self) -> None:
        self.model: Optional[object] = None
        self.use_yolo = False
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(config.YOLO_MODEL)
            self.use_yolo = True
            print("[INFO] YOLOv8 loaded for player detection.")
        except Exception as exc:
            print(f"[WARN] YOLOv8 not available ({exc}). Using OpenCV fallback detector.")

    def detect(self, frame: np.ndarray) -> List[Player]:
        if self.use_yolo and self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_fallback(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Player]:
        results = self.model.predict(frame, classes=[0], conf=config.CONFIDENCE, verbose=False)
        players: list[Player] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                area = (x2 - x1) * (y2 - y1)
                if area < 500:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                players.append(Player("", (x1, y1, x2, y2), (cx, cy)))
        players = sorted(players, key=lambda p: p.center[1], reverse=True)[: config.MAX_PLAYERS]
        for i, p in enumerate(players, start=1):
            p.player_id = f"Player_{i}"
        return players

    def _detect_fallback(self, frame: np.ndarray) -> List[Player]:
        """Detect non-blue objects on the court. This is not perfect but works as a backup."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Blue court mask; players are mostly not blue.
        blue = cv2.inRange(hsv, (85, 40, 40), (130, 255, 255))
        non_blue = cv2.bitwise_not(blue)
        # Focus on central court area; ignore upper timestamp and outside borders.
        h, w = frame.shape[:2]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[int(0.05*h):int(0.90*h), int(0.05*w):int(0.95*w)] = 255
        mask = cv2.bitwise_and(non_blue, roi_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if 250 < area < 8000 and 10 < bw < 120 and 20 < bh < 160:
                candidates.append((x, y, x + bw, y + bh, area))
        candidates = sorted(candidates, key=lambda b: b[4], reverse=True)[: config.MAX_PLAYERS]
        players = []
        for i, (x1, y1, x2, y2, _) in enumerate(candidates, start=1):
            players.append(Player(f"Player_{i}", (x1, y1, x2, y2), ((x1+x2)//2, (y1+y2)//2)))
        return players
