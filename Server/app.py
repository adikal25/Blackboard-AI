from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2 as cv
from detect_gestures import ActionDetector
from draw import Draw
import threading
import queue

app = Flask(__name__)
CORS(app)

# Global variables
frame_queue = queue.Queue(maxsize=10)
action_queue = queue.Queue(maxsize=10)

def gen_frame():
    while True:
        frame = frame_queue.get()
        _, buffer = cv.imencode('.jpg',frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_action')
def get_action():
    action = action_queue.get()
    return jsonify(({"action":action}))
def process_frames():
    cap=cv.VideoCapture(1)
    detector=ActionDetector()
    draw=None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)

        if draw is None:
            draw = Draw(frame.shape)

        processed_frame, action = detector.detect_action(frame)
        output_frame = draw.draw(processed_frame, action,
                                            detector.prev_hand_landmarks,
                                            detector.mp_hands)

        if action:
            cv.putText(output_frame, f"Action: {action}", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if not frame_queue.full():
            frame_queue.put(output_frame)
        if not action_queue.full():
            action_queue.put(action)

    cap.release()


    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()

