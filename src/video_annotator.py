from __future__ import annotations
from collections import Counter
import cv2
from src.ball_tracker import BallState
from src.player_detector import Player
from src.shot_classifier import ShotPrediction


def draw_annotations(frame, players: list[Player], ball: BallState | None, last_shot: ShotPrediction | None, counts: Counter):
    for p in players:
        x1, y1, x2, y2 = p.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(frame, p.player_id, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
        # Probable racket/swing zone around player - prototype approximation.
        cx, cy = p.center
        cv2.circle(frame, (cx, cy), 60, (255, 180, 0), 1)

    if ball is not None:
        cv2.circle(frame, ball.center, max(4, ball.radius + 2), (0, 255, 255), 2)
        cv2.putText(frame, "ball", (ball.center[0] + 8, ball.center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    y = 25
    cv2.rectangle(frame, (8, 8), (380, 120), (0, 0, 0), -1)
    cv2.putText(frame, "Padel Shot Analytics Prototype", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 26
    if last_shot is not None:
        msg = f"Last shot: {last_shot.player_id} - {last_shot.shot_type} @ {last_shot.timestamp_sec:.1f}s"
    else:
        msg = "Last shot: detecting..."
    cv2.putText(frame, msg, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 24
    cv2.putText(frame, f"Counts: {dict(counts)}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return frame
