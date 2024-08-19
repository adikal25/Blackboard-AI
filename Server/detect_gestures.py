import cv2 as cv
import mediapipe as mp
import numpy as np
from draw import Draw

class ActionDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.actions = ["draw", "erase", "clear"]
        self.prev_hand_landmarks = None
        self.draw = None

    def detect_action(self, frame):
        if self.draw is None:
            self.draw = Draw(frame.shape)

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        action = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Detect action based on hand pose
            action = self._classify_action(hand_landmarks)

            # Update previous landmarks
            self.prev_hand_landmarks = hand_landmarks
        frame = self.draw.draw(frame, action, hand_landmarks if results.multi_hand_landmarks else None, self.mp_hands)

        return frame, action

    def _classify_action(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        if self._is_pinch(index_tip, thumb_tip):
            return "draw"
        if self._is_open_palm(hand_landmarks):
            return "erase"
        if self._all_fingersclose(hand_landmarks):
            return "clear"
        return None

    def _is_pinch(self, index_tip, thumb_tip):
        distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
        return distance < 1/20  # Threshold for pinch

    def _is_open_palm(self, hand_landmarks):
        fingertips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        return all(hand_landmarks.landmark[tip].y < wrist.y for tip in fingertips)

    def _all_fingersclose(self, hand_landmarks):
        fingertips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.sqrt((fingertips[i].x - fingertips[j].x) ** 2 + (fingertips[i].y - fingertips[j].y) ** 2)
                if dist > 0.25:
                    return False
        return True
