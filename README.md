# Padel Game Analytics — Shot Classification System

## Project Overview

This project is a Computer Vision-based prototype that analyzes padel gameplay videos to detect players, track the ball, and classify basic shot types such as forehand, backhand, and smash.

The goal of this assignment is to demonstrate practical application of AI/ML concepts in a real-world sports analytics scenario.

---

##  Objectives

The system performs the following tasks:

- Detect players in the video
- Track the ball across frames
- Identify shot events
- Classify shots into:
  - Forehand
  - Backhand
  - Smash / Serve
- Output structured results in CSV and JSON format

---

## Approach & Methodology

### 1. Player Detection
- Used a pretrained YOLOv8 model to detect players in each frame
- Bounding boxes are used to track player positions

### 2. Ball Tracking
- Implemented using OpenCV
- Color-based detection (HSV) to identify the ball
- Contour detection used to track ball position frame-by-frame

### 3. Shot Detection
- A shot event is detected when:
  - The ball is close to a player
  - There is a sudden change in ball direction or speed

### 4. Shot Classification (Rule-Based)
- Forehand → Ball on right side of player
- Backhand → Ball on left side of player
- Smash/Serve → Ball contact at higher position with higher outgoing speed

---

##  Tech Stack

- Python
- OpenCV
- Ultralytics YOLOv8 (pretrained)
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

padel-shot-classification/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── player_detector.py
│   ├── ball_tracker.py
│   ├── shot_classifier.py
│   └── utils.py
├── output/
│   ├── shot_predictions.csv
│   ├── shot_predictions.json
│   └── shot_summary.png


---

##  How to Run

### 1. Install dependencies
pip3 install -r requirements.txt


### 2. Add input video
Place your video inside: input/sample_video.mp4


### 3. Run the program 
python3 main.py --video input/sample_video.mp4 --output output


---

## Output

The system generates:

- `shot_predictions.csv` → structured shot data  
- `shot_predictions.json` → JSON format output  
- `shot_summary.png` → basic visualization  
- `annotated_video.mp4` → processed video with detections (shared separately)

---

## Demo Video

👉 Add your Google Drive link here

---

##  Limitations

- Ball detection may fail during fast motion or occlusion  
- Racket is not explicitly detected (approximated using motion)  
- Shot classification is rule-based, not learned  
- Accuracy depends on video quality and camera angle  

---

## Future Improvements

- Train custom model for ball and racket detection  
- Use deep learning for shot classification  
- Improve player tracking with consistent IDs  
- Use trajectory-based analysis for better accuracy  
- Build real-time analytics dashboard  

---

## Key Learning

This project demonstrates:

- Practical Computer Vision pipeline design  
- Integration of detection, tracking, and classification  
- Use of pretrained models for rapid prototyping  
- Importance of simple and explainable solutions  

---

## Keywords

Padel, Computer Vision, Shot Classification, Sports Analytics, YOLO, OpenCV

---

## Author

Bikash Yadav

