from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json, pathlib, threading, os
from eye_tracking import process_eye_coverage

app = Flask(__name__)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Eye aspect ratio calculation
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

# Gaze estimation
def estimate_gaze(iris, eye_points):
    eye_center = np.mean(eye_points, axis=0)
    eye_width = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    eye_height = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    dx = (iris[0] - eye_center[0]) / eye_width
    dy = (iris[1] - eye_center[1]) / eye_height
    if dx < -0.25: return "Left"
    elif dx > 0.25: return "Right"
    elif dy < -0.3: return "Up"
    elif dy > 0.3: return "Down"
    else: return "Center"

# Attention status
def get_attention_status(left_gaze, right_gaze):
    if left_gaze in ["Left", "Right"] or right_gaze in ["Left", "Right"]:
        return "Not Attentive - Looking Away"
    elif left_gaze == "Center" and right_gaze == "Center":
        return "Attentive - Focused"
    else:
        return "Partially Attentive"
def analyze_gaze_direction(left_gaze, right_gaze):
    if left_gaze == right_gaze:
        if left_gaze == "Center": return "Looking Center", "Focused"
        if left_gaze == "Left": return "Looking Left", "Distracted Left"
        if left_gaze == "Right": return "Looking Right", "Distracted Right"
        if left_gaze == "Up": return "Looking Up", "Looking Up"
        if left_gaze == "Down": return "Looking Down", "Looking Down"
    return "Mixed Gaze", "Eyes Not Aligned"

# Estimate eye coverage
def estimate_eye_coverage(image, eye_points):
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
    elif coverage_percentage <= 60:
        return "50% covered"
    elif coverage_percentage <= 80:
        return "75% covered"
    else:
        return "100% covered"

# Get coverage status
def get_coverage_status(left_coverage, right_coverage):
    if left_coverage == "0% covered" and right_coverage == "0% covered":
        return "No eyes covered"
    elif left_coverage != "0% covered" and right_coverage == "0% covered":
        return f"Left eye: {left_coverage}"
    elif right_coverage != "0% covered" and left_coverage == "0% covered":
        return f"Right eye: {right_coverage}"
    else:
        return f"Left eye: {left_coverage}, Right eye: {right_coverage}"

# Process image
def process_image(image):
    frame = cv2.flip(image, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    response = {
        "status": "success",
        "left_eye_status": None,
        "right_eye_status": None,
        "left_gaze": None,
        "right_gaze": None,
        "left_coverage": None,
        "right_coverage": None,
        "attention": None,
        "coverage_status": None,
        "gaze_direction": None,
        "gaze_message": None,
        "annotated_image": None
    }

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Landmark indices
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        left_iris_index = 468
        right_iris_index = 473

        # Get landmarks in image space
        left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
        right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]
        left_iris = (int(face_landmarks.landmark[left_iris_index].x * w), int(face_landmarks.landmark[left_iris_index].y * h))
        right_iris = (int(face_landmarks.landmark[right_iris_index].x * w), int(face_landmarks.landmark[right_iris_index].y * h))

        # Analyze status
        left_status = detect_eye_status(left_eye)
        right_status = detect_eye_status(right_eye)

        left_gaze = estimate_gaze(left_iris, left_eye)
        right_gaze = estimate_gaze(right_iris, right_eye)

        left_coverage = estimate_eye_coverage(frame, left_eye)
        right_coverage = estimate_eye_coverage(frame, right_eye)

        attention = get_attention_status(left_gaze, right_gaze)
        coverage_status = get_coverage_status(left_coverage, right_coverage)
        gaze_direction, gaze_message = analyze_gaze_direction(left_gaze, right_gaze)

        # Update response
        response.update({
            "left_eye_status": left_status,
            "right_eye_status": right_status,
            "left_gaze": left_gaze,
            "right_gaze": right_gaze,
            "left_coverage": left_coverage,
            "right_coverage": right_coverage,
            "attention": attention,
            "coverage_status": coverage_status,
            "gaze_direction": gaze_direction,
            "gaze_message": gaze_message
        })

        # Annotate image
        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
        cv2.circle(frame, left_iris, 2, (255, 0, 255), -1)
        cv2.circle(frame, right_iris, 2, (255, 0, 255), -1)

        # Convert annotated frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        response["annotated_image"] = f"data:image/jpeg;base64,{frame_base64}"

    else:
        response["status"] = "error"
        response["message"] = "No face detected"

    return response


# Path to results JSON (env override allowed)
DEFAULT_RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "eye_tracking_results.json"
RESULTS_PATH = pathlib.Path(os.getenv("EYE_RESULTS_PATH", str(DEFAULT_RESULTS_PATH)))

# -------------------------------------------------------------
# Start eye-tracking capture loop in a background thread 
# -------------------------------------------------------------
def _start_capture_once():
    """Start webcam capture exactly once to avoid double spawn on Flask reloads."""
    if getattr(app, "_capture_started", False):
        return
    # Prevent duplicate start when using Flask auto-reload
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not os.environ.get("FLASK_ENV"):
        t = threading.Thread(target=process_eye_coverage, args=(0,), kwargs={"output_path": str(RESULTS_PATH)}, daemon=True)
        t.start()
        app._capture_started = True

_start_capture_once()

# Eye-tracking metrics endpoint (API-only)
@app.route('/eye_metrics', methods=['GET'])
def eye_metrics():
    try:
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        latest = data[-1] if data else {}
        return jsonify(latest)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if 'image' not in request.json:
            return jsonify({"status": "error", "message": "No image provided"}), 400
        
        # Decode base64 image
        img_data = base64.b64decode(request.json['image'].split(',')[1])
        img = Image.open(BytesIO(img_data))
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        response = process_image(frame)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
        
        
        
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "success", "message": "Hello, World!"})
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
