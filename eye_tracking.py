import cv2
import mediapipe as mp
import numpy as np
import math
from collections import Counter
import json
from datetime import datetime
from pathlib import Path

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

# Eye aspect ratio calculation (for open/shut detection)
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Detect eye open/shut
def detect_eye_status(eye_points):
    ear = eye_aspect_ratio(eye_points)
    return "Closed" if ear < 0.25 else "Open"

# Gaze estimation using iris and eye landmarks
def estimate_gaze(iris, eye_points):
    eye_center = np.mean(eye_points, axis=0)
    eye_width = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    eye_height = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))

    # Normalize iris position relative to eye center
    dx = (iris[0] - eye_center[0]) / eye_width
    dy = (iris[1] - eye_center[1]) / eye_height

    # Tune these thresholds as needed
    if dx < -0.25:
        return "Left"
    elif dx > 0.25:
        return "Right"
    elif dy < -0.3:
        return "Up"
    elif dy > 0.3:
        return "Down"
    else:
        return "Center"

# Enhanced attention status based on gaze direction
def is_attentive(left_gaze, right_gaze):
    # User is attentive only when both eyes are looking center
    # Left or right gaze indicates distraction
    attentive_gazes = ["Center", "Up", "Down"]  # Up and Down are still considered attentive
    not_attentive_gazes = ["Left", "Right"]
    
    left_attentive = left_gaze in attentive_gazes
    right_attentive = right_gaze in attentive_gazes
    
    return left_attentive and right_attentive

# Get detailed attention status
def get_attention_status(left_gaze, right_gaze):
    # Check if either eye is looking left or right - this means not attentive
    if left_gaze in ["Left", "Right"] or right_gaze in ["Left", "Right"]:
        return "Not Attentive - Looking Away"
    elif left_gaze == "Center" and right_gaze == "Center":
        return "Attentive - Focused"
    else:
        return "Partially Attentive"

# Check specifically for left/right gaze
def is_looking_away(left_gaze, right_gaze):
    """Check if person is looking left or right (not attentive)"""
    return left_gaze in ["Left", "Right"] or right_gaze in ["Left", "Right"]

# Track gaze duration
def track_gaze_duration(current_gaze, gaze_history, max_duration=30):
    """Track how long someone has been looking in a specific direction"""
    gaze_history.append(current_gaze)
    if len(gaze_history) > max_duration:
        gaze_history.pop(0)
    
    # Count consecutive frames of looking away
    away_count = 0
    for gaze in reversed(gaze_history):
        if gaze in ["Left", "Right"]:
            away_count += 1
        else:
            break
    
    return away_count, len(gaze_history)

# Get detailed gaze analysis
def analyze_gaze_direction(left_gaze, right_gaze):
    """Provides detailed analysis of gaze direction"""
    if left_gaze == "Left" and right_gaze == "Left":
        return "Looking Left", "Distracted - Looking Left"
    elif left_gaze == "Right" and right_gaze == "Right":
        return "Looking Right", "Distracted - Looking Right"
    elif left_gaze == "Center" and right_gaze == "Center":
        return "Looking Center", "Focused - Looking Forward"
    elif left_gaze == "Up" and right_gaze == "Up":
        return "Looking Up", "Looking Up"
    elif left_gaze == "Down" and right_gaze == "Down":
        return "Looking Down", "Looking Down"
    else:
        return "Mixed Gaze", "Eyes Not Aligned"

# Eye coverage function remains unchanged
def estimate_eye_coverage(image, eye_points):
    print(f"eye points: {eye_points}")
    eye_region = np.array(eye_points, dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye_area = cv2.bitwise_and(image, image, mask=mask)
    gray_eye = cv2.cvtColor(eye_area, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_eye, 150, 255, cv2.THRESH_BINARY)
    total_pixels = np.sum(mask == 255)
    covered_pixels = np.sum(thresh == 255)
    covered_pixels = min(covered_pixels, total_pixels)
    coverage_percentage = (covered_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    if coverage_percentage < 10:
        return "0% covered"
    elif coverage_percentage < 75:
        return "50% covered"
    else:
        return "100% covered"

# Determine eye coverage status
def get_coverage_status(left_coverage, right_coverage):
    if left_coverage == "0% covered" and right_coverage == "0% covered":
        return "No eyes covered"
    elif left_coverage != "0% covered" and right_coverage == "0% covered":
        return f"Left eye: {left_coverage}"
    elif right_coverage != "0% covered" and left_coverage == "0% covered":
        return f"Right eye: {right_coverage}"
    else:
        return f"Left eye: {left_coverage}, Right eye: {right_coverage}"

# Process camera input
def process_eye_coverage(input_source, output_path=None):
    cap = cv2.VideoCapture(input_source if isinstance(input_source, str) else 0)
    if not cap.isOpened():
        print("Error: Could not open input source.")
        return
    
    # Gaze history for stability
    gaze_history = []
    history_length = 3  # Number of frames to consider for stable detection
    
    # Gaze duration tracking
    gaze_duration_history = []
    away_duration = 0
    
    # Store per-frame analysis for JSON export
    results_data = []
    frame_index = 0
    flush_every = 30  # flush to disk every 30 frames (~1 sec at 30 FPS)
    
    # Resolve output path (face_detectionmlkit_final directory)
    if output_path is None:
        data_dir = Path(__file__).resolve().parent 
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "eye_tracking_results.json"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmark indices
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                left_iris_index = 468
                right_iris_index = 473

                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]
                left_iris = (int(face_landmarks.landmark[left_iris_index].x * w), int(face_landmarks.landmark[left_iris_index].y * h))
                right_iris = (int(face_landmarks.landmark[right_iris_index].x * w), int(face_landmarks.landmark[right_iris_index].y * h))

                # Coverage
                left_coverage = estimate_eye_coverage(frame, left_eye)
                right_coverage = estimate_eye_coverage(frame, right_eye)
                coverage_status = get_coverage_status(left_coverage, right_coverage)

                # Eye open/shut
                left_status = detect_eye_status(left_eye)
                right_status = detect_eye_status(right_eye)

                # Gaze estimation
                left_gaze = estimate_gaze(left_iris, left_eye)
                right_gaze = estimate_gaze(right_iris, right_eye)
                
                # Add to gaze history
                current_gaze = (left_gaze, right_gaze)
                gaze_history.append(current_gaze)
                if len(gaze_history) > history_length:
                    gaze_history.pop(0)
                
                # Get stable gaze direction
                if len(gaze_history) == history_length:
                    # Check if gaze is consistent
                    left_gazes = [g[0] for g in gaze_history]
                    right_gazes = [g[1] for g in gaze_history]
                    
                    # Use most common gaze direction for stability
                    from collections import Counter
                    left_stable_gaze = Counter(left_gazes).most_common(1)[0][0]
                    right_stable_gaze = Counter(right_gazes).most_common(1)[0][0]
                    
                    left_gaze = left_stable_gaze
                    right_gaze = right_stable_gaze

                attention = get_attention_status(left_gaze, right_gaze)
                gaze_direction, gaze_message = analyze_gaze_direction(left_gaze, right_gaze)
                looking_away = is_looking_away(left_gaze, right_gaze)

                # Collect data for JSON output
                results_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "left_eye_status": left_status,
                    "right_eye_status": right_status,
                    "left_gaze": left_gaze,
                    "right_gaze": right_gaze,
                    "attention": attention,
                    "gaze_direction": gaze_direction,
                    "coverage_status": coverage_status
                })

                # Periodically flush to disk so API sees near real-time data
                frame_index += 1
                if frame_index % flush_every == 0:
                    try:
                        with open(output_path, "w") as f:
                            json.dump(results_data, f, indent=2)
                    except Exception as e:
                        print(f"Error flushing results: {e}")

                # Draw eye contours
                cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
                cv2.circle(frame, left_iris, 2, (255, 0, 255), -1)
                cv2.circle(frame, right_iris, 2, (255, 0, 255), -1)

                # Draw gaze direction indicators
                gaze_color = (0, 255, 0) if attention.startswith("Attentive") else (0, 0, 255)
                
                # Display status with improved formatting
                cv2.putText(frame, f"Coverage: {coverage_status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.putText(frame, f"Left Eye: {left_status} | Gaze: {left_gaze}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"Right Eye: {right_status} | Gaze: {right_gaze}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(frame, f"Gaze Direction: {gaze_direction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"Attention: {attention}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_color, 2)
                
                # Enhanced gaze direction warning
                if looking_away:
                    cv2.putText(frame, "LOOKING AWAY!", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    cv2.putText(frame, gaze_message, (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Add specific direction indicator
                    if left_gaze == "Left" or right_gaze == "Left":
                        cv2.putText(frame, "LOOKING LEFT", (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif left_gaze == "Right" or right_gaze == "Right":
                        cv2.putText(frame, "LOOKING RIGHT", (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Eye & Gaze Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Final flush on exit
    try:
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    process_eye_coverage(0)
