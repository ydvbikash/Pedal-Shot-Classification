from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

import config


@dataclass
class BallState:
    frame: int
    center: tuple[int, int]
    radius: int


class BallTracker:
    """HSV-based ball tracker for yellow/green padel ball."""

    def __init__(self) -> None:
        self.last_ball: Optional[BallState] = None
        self.prev_gray: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray, frame_index: int) -> Optional[BallState]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(config.BALL_HSV_LOWER, dtype=np.uint8)
        upper = np.array(config.BALL_HSV_UPPER, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1
        for c in contours:
            area = cv2.contourArea(c)
            if not (config.BALL_MIN_AREA <= area <= config.BALL_MAX_AREA):
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < 1 or radius > 12:
                continue
            perimeter = cv2.arcLength(c, True)
            circularity = 0 if perimeter == 0 else 4 * np.pi * area / (perimeter ** 2)
            # Prefer circular, small bright objects. If last ball exists, prefer nearby candidates.
            score = circularity * 10 - radius
            if self.last_ball:
                dx = x - self.last_ball.center[0]
                dy = y - self.last_ball.center[1]
                score -= 0.01 * (dx*dx + dy*dy) ** 0.5
            if score > best_score:
                best_score = score
                best = BallState(frame_index, (int(x), int(y)), int(radius))

        if best is None:
            best = self._detect_motion_proxy(frame, frame_index)

        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if best is not None:
            self.last_ball = best
        return best

    def _detect_motion_proxy(self, frame: np.ndarray, frame_index: int) -> Optional[BallState]:
        """Fallback when color detection misses the ball: choose a small moving blob.

        This is a proxy tracker, not perfect ball detection. It helps the prototype keep
        working when the ball is blurred or lighting changes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            return None
        diff = cv2.absdiff(gray, self.prev_gray)
        _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        best = None
        best_score = -1e9
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if not (4 <= area <= 140):
                continue
            if y < int(0.03*h) or y > int(0.92*h) or x < int(0.03*w) or x > int(0.97*w):
                continue
            cx, cy = x + bw // 2, y + bh // 2
            score = 100 - area
            if self.last_ball:
                score -= 0.08 * ((cx-self.last_ball.center[0])**2 + (cy-self.last_ball.center[1])**2) ** 0.5
            if score > best_score:
                best_score = score
                best = BallState(frame_index, (int(cx), int(cy)), max(2, int(max(bw, bh)/2)))
        return best
