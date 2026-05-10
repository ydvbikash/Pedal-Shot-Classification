from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
import math

import config
from src.ball_tracker import BallState
from src.player_detector import Player
from src.utils import angle_between, distance, side_of_player


@dataclass
class ShotPrediction:
    frame: int
    timestamp_sec: float
    player_id: str
    shot_type: str
    ball_x: int
    ball_y: int
    nearest_player_distance: float
    ball_speed: float
    direction_change_deg: float

    def to_dict(self):
        d = asdict(self)
        d["timestamp_sec"] = round(d["timestamp_sec"], 2)
        d["nearest_player_distance"] = round(d["nearest_player_distance"], 2)
        d["ball_speed"] = round(d["ball_speed"], 2)
        d["direction_change_deg"] = round(d["direction_change_deg"], 2)
        return d


class ShotClassifier:
    """Rule-based shot event detector and classifier.

    A shot is detected when the tracked ball is near a player and its speed/direction changes.
    This keeps the prototype explainable and avoids needing a custom labelled dataset.
    """

    def __init__(self, fps: float, frame_height: int) -> None:
        self.fps = fps
        self.frame_height = frame_height
        self.ball_history: List[BallState] = []
        self.last_shot_frame = -9999

    def update(self, ball: Optional[BallState], players: List[Player]) -> Optional[ShotPrediction]:
        if ball is None:
            return None
        self.ball_history.append(ball)
        self.ball_history = self.ball_history[-8:]
        if len(self.ball_history) < 4 or not players:
            return None

        if ball.frame - self.last_shot_frame < config.SHOT_COOLDOWN_FRAMES:
            return None

        b0, b1, b2 = self.ball_history[-3], self.ball_history[-2], self.ball_history[-1]
        v_prev = (b1.center[0] - b0.center[0], b1.center[1] - b0.center[1])
        v_now = (b2.center[0] - b1.center[0], b2.center[1] - b1.center[1])
        speed = math.hypot(v_now[0], v_now[1])
        change = angle_between(v_prev, v_now)

        nearest = min(players, key=lambda p: distance(ball.center, p.center))
        d = distance(ball.center, nearest.center)

        near_player = d <= config.PLAYER_BALL_DISTANCE
        motion_event = speed >= config.MIN_BALL_SPEED or change >= config.DIRECTION_CHANGE_THRESHOLD
        if not (near_player and motion_event):
            return None

        shot_type = self._classify(ball, nearest, speed)
        self.last_shot_frame = ball.frame
        return ShotPrediction(
            frame=ball.frame,
            timestamp_sec=ball.frame / self.fps,
            player_id=nearest.player_id,
            shot_type=shot_type,
            ball_x=ball.center[0],
            ball_y=ball.center[1],
            nearest_player_distance=d,
            ball_speed=speed,
            direction_change_deg=change,
        )

    def _classify(self, ball: BallState, player: Player, speed: float) -> str:
        high_contact = ball.center[1] < self.frame_height * config.SMASH_Y_RATIO
        if high_contact and speed >= config.SMASH_SPEED:
            return "smash_or_serve"
        side = side_of_player(ball.center[0], player.center[0])
        # Assumption: in this fixed overhead camera, ball on player's right side is forehand.
        # This is a prototype simplification; handedness/body pose would improve it.
        return "forehand" if side == "right" else "backhand"
