# Padel Game Analytics — Shot Classification System

This project is a simple, explainable computer-vision prototype which analyzes padel gameplay video, detects/tracks players and the ball, estimates racket/swing activity near players, classifies basic shot types, and exports CSV/JSON results.

## Why this approach?

The assignment asks for a working prototype, not a perfect production system. Training a deep-learning shot classifier requires labelled padel data, GPU time, and more than a few days. Therefore, this project uses:

- **YOLOv8 pretrained model** for player detection.
- **OpenCV HSV color tracking** for the yellow/green ball.
- **Rule-based shot detection** using ball-player distance, ball speed, and direction change.
- **Simple rule-based shot classification** into forehand, backhand, and smash/serve.

This makes the system practical, understandable, and easy to explain in an internship interview.

## Features

Mandatory tasks:

- Detect and track players.
- Detect and track ball.
- Estimate racket/swing area near players using motion/region approximation.
- Classify at least 2–3 shots:
  - Forehand
  - Backhand
  - Smash/Serve
- Export structured output:
  - `shot_predictions.csv`
  - `shot_predictions.json`

Bonus tasks:

- Annotated output video.
- Shot count analytics chart.
- Rule-based logic for shot events.

## Project Structure

```text
padel-shot-classification/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── src/
│   ├── player_detector.py
│   ├── ball_tracker.py
│   ├── shot_classifier.py
│   ├── video_annotator.py
│   └── utils.py
├── input/
│   └── sample_video.mp4
└── output/
    ├── annotated_video.mp4
    ├── shot_predictions.csv
    ├── shot_predictions.json
    └── shot_summary.png
```

## Setup

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Place the sample video inside the `input/` folder:

```text
input/sample_video.mp4
```

Run the full pipeline:

```bash
python main.py --video input/sample_video.mp4 --output output
```

For a quick test on the first 1500 frames:

```bash
python main.py --video input/sample_video.mp4 --output output --max-frames 1500
```

## Output Format

Example CSV/JSON fields:

| Column | Meaning |
|---|---|
| frame | Video frame where shot was detected |
| timestamp_sec | Timestamp in seconds |
| player_id | Nearest player at contact moment |
| shot_type | forehand / backhand / smash_or_serve |
| ball_x, ball_y | Ball location in video frame |
| nearest_player_distance | Distance from ball to nearest player |
| ball_speed | Ball movement speed between frames |
| direction_change_deg | Direction change angle of ball movement |

## Methodology

### 1. Player Detection

The system first detects people in each video frame. The preferred method is YOLOv8 pretrained on the COCO dataset. Since the model already knows the `person` class, no custom training is required.

### 2. Ball Tracking

The padel ball is small and can be difficult for a general object detector. Instead, the system uses HSV color segmentation to find yellow/green circular objects. It filters candidates by area, radius, circularity, and closeness to the previous ball location.

### 3. Racket/Swing Approximation

The racket is very small in a wide camera view. Instead of pretending the system can perfectly detect it, this prototype estimates a probable racket/swing region around each player. The annotated video shows this region. In a future version, this can be replaced by a custom trained racket detector.

### 4. Shot Event Detection

A shot is detected when:

```text
ball is near a player
AND
ball speed or direction changes suddenly
```

This is based on the idea that racket contact changes the ball trajectory.

### 5. Shot Classification

The prototype classifies shots using simple rules:

```text
High contact point + high speed → smash_or_serve
Ball on player's right side → forehand
Ball on player's left side → backhand
```

This is an explainable baseline. It is not perfect, but it is suitable for a short internship assignment prototype.

## Challenges Faced

- The ball is very small and sometimes blends with the court or lighting.
- The racket is difficult to detect from a wide CCTV-style camera angle.
- Player pose and handedness are not available, so forehand/backhand classification is approximate.
- Occlusion by glass/net/other players may cause missing ball detections.

## Future Improvements

- Train a custom YOLO model for padel ball and racket detection.
- Use pose estimation with MediaPipe to identify body orientation and racket-hand side.
- Use multi-object tracking such as ByteTrack or DeepSORT for stable player IDs.
- Add bounce detection using ball vertical motion and court-line calibration.
- Build a small Streamlit dashboard for shot analytics.

## Demo Files to Submit

Submit these files/folders:

1. GitHub repository with this code.
2. `output/annotated_video.mp4` as demo.
3. `output/shot_predictions.csv` and `output/shot_predictions.json`.
4. README explanation.
5. If YOLO downloads `yolov8n.pt`, upload it to Google Drive or mention that it auto-downloads from Ultralytics on first run.
