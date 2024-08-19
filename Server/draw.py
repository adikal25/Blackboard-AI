import cv2 as cv
import numpy as np

class Draw:
    def __init__(self, frame_shape):
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)
        self.drawing_color = (0, 255, 0)
        self.eraser_color = (0, 0, 0)
        self.drawing_thickness = 5
        self.eraser_thickness = 20
        self.prev_point = None
        self.smoothing_factor = 0.2

    def draw(self, frame, action, hand_landmarks, mp_hands):
        if hand_landmarks:
            current_point = self._get_drawing_point(frame, hand_landmarks, mp_hands)

            if action == "draw":
                self._draw_line(current_point, self.drawing_color, self.drawing_thickness)
            elif action == "erase":
                self._draw_line(current_point, self.eraser_color, self.eraser_thickness)
            elif action == "clear":
                self.clear()

            self.prev_point = current_point
        else:
            self.prev_point = None

        return self.get_output_frame(frame)

    def _get_drawing_point(self, frame, hand_landmarks, mp_hands):
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        h, w, _ = frame.shape
        x, y = int(index_tip.x * w), int(index_tip.y * h)

        if self.prev_point:
            x = int(self.prev_point[0] + self.smoothing_factor * (x - self.prev_point[0]))
            y = int(self.prev_point[1] + self.smoothing_factor * (y - self.prev_point[1]))

        return x, y

    def _draw_line(self, current_point, color, thickness):
        if self.prev_point:
            cv.line(self.canvas, self.prev_point, current_point, color, thickness)

    def clear(self):
        self.canvas.fill(0)

    def get_output_frame(self, frame):
        return cv.addWeighted(frame, 1, self.canvas, 0.5, 0)

    def set_drawing_thickness(self, thickness):
        self.drawing_thickness = thickness

    def set_smoothing_factor(self, factor):
        self.smoothing_factor = max(0, min(1, factor))

    def set_drawing_color(self, color):
        self.drawing_color = color
