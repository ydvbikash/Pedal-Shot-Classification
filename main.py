from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.ball_tracker import BallTracker
from src.player_detector import PlayerDetector
from src.shot_classifier import ShotClassifier
from src.video_annotator import draw_annotations


def resize_keep_aspect(frame, width: int):
    h, w = frame.shape[:2]
    if width is None or w == width:
        return frame
    new_h = int(h * width / w)
    return cv2.resize(frame, (width, new_h))


def run(video_path: str, output_dir: str, max_frames: int | None = None, frame_skip: int | None = None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    process_fps = original_fps / (frame_skip or config.FRAME_SKIP)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame.")
    frame = resize_keep_aspect(frame, config.RESIZE_WIDTH)
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output / "annotated_video.mp4"), fourcc, process_fps, (width, height))

    player_detector = PlayerDetector()
    ball_tracker = BallTracker()
    shot_classifier = ShotClassifier(fps=original_fps, frame_height=height)

    predictions = []
    counts = Counter()
    last_shot = None
    last_players = []
    frame_index = 0
    processed = 0

    # process the already-read first frame by rewinding for cleaner loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    skip = frame_skip or config.FRAME_SKIP
    max_frames = max_frames if max_frames is not None else config.MAX_FRAMES

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_index >= max_frames:
            break
        if frame_index % skip != 0:
            frame_index += 1
            continue

        frame = resize_keep_aspect(frame, config.RESIZE_WIDTH)
        # Player detection is more expensive than ball tracking; refresh it every few processed frames.
        if processed % 5 == 0 or not last_players:
            last_players = player_detector.detect(frame)
        players = last_players
        ball = ball_tracker.detect(frame, frame_index)
        shot = shot_classifier.update(ball, players)
        if shot is not None:
            last_shot = shot
            counts[shot.shot_type] += 1
            predictions.append(shot.to_dict())

        annotated = draw_annotations(frame.copy(), players, ball, last_shot, counts)
        video_writer.write(annotated)

        processed += 1
        if processed % 100 == 0:
            print(f"[INFO] processed frames: {processed}, shots: {len(predictions)}")
        frame_index += 1

    cap.release()
    video_writer.release()

    df = pd.DataFrame(predictions)
    if df.empty:
        df = pd.DataFrame(columns=[
            "frame", "timestamp_sec", "player_id", "shot_type", "ball_x", "ball_y",
            "nearest_player_distance", "ball_speed", "direction_change_deg"
        ])
    df.to_csv(output / "shot_predictions.csv", index=False)
    with open(output / "shot_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # simple analytics chart
    plt.figure(figsize=(7, 4))
    if counts:
        plt.bar(list(counts.keys()), list(counts.values()))
        plt.title("Detected Shot Counts")
        plt.xlabel("Shot Type")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No shots detected - tune thresholds", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output / "shot_summary.png", dpi=160)
    plt.close()

    print("\n[DONE] Files saved:")
    print(output / "annotated_video.mp4")
    print(output / "shot_predictions.csv")
    print(output / "shot_predictions.json")
    print(output / "shot_summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Padel shot classification prototype")
    parser.add_argument("--video", required=True, help="Path to input padel video")
    parser.add_argument("--output", default="output", help="Output folder")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick demo")
    parser.add_argument("--frame-skip", type=int, default=None, help="Process every Nth frame")
    args = parser.parse_args()
    run(args.video, args.output, args.max_frames, args.frame_skip)
