from __future__ import annotations
import math
from typing import Tuple

Point = Tuple[int, int]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_between(v1, v2) -> float:
    """Return angle in degrees between two 2D vectors."""
    x1, y1 = v1
    x2, y2 = v2
    n1 = math.hypot(x1, y1)
    n2 = math.hypot(x2, y2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = x1 * x2 + y1 * y2
    cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_theta))


def side_of_player(ball_x: int, player_center_x: int) -> str:
    return "right" if ball_x >= player_center_x else "left"
