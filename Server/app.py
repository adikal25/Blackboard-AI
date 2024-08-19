from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2 as cv
from detect_gestures import ActionDetector
from draw import Draw
import threading
import queue
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
frame_queue = queue.Queue(maxsize=100000000000000000)
action_queue = queue.Queue(maxsize=100000000000000000)

def gen_frame():
    while True:
        frame = frame_queue.get()
        _, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return 'Hello World!'

@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_action')
def get_action():
    if not action_queue.empty():
        action = action_queue.get()
        return jsonify({"action": action})
    return jsonify({"action": None})

def process_frames():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        app.logger.error("Failed to open camera")
        return

    detector = ActionDetector()
    draw = None

    while True:
        if not cap.isOpened():
            app.logger.info("Reopening camera...")
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                app.logger.error("Failed to reopen camera, retrying in 5 seconds")
                time.sleep(5)
                continue

        ret, frame = cap.read()
        if not ret:
            app.logger.warning("Failed to capture image, attempting to reopen camera")
            cap.release()
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                app.logger.error("Failed to reopen camera")
                break
            continue

        frame = cv.flip(frame, 1)

        if draw is None:
            draw = Draw(frame.shape)

        processed_frame, action = detector.detect_action(frame)
        output_frame = draw.draw(processed_frame, action,
                                detector.prev_hand_landmarks,
                                detector.mp_hands)

        if not frame_queue.full():
            frame_queue.put(output_frame)
        if not action_queue.full():
            action_queue.put(action)

    cap.release()

if __name__ == '__main__':
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
