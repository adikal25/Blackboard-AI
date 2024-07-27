import cv2
from detect_gestures import ActionDetector
from draw import Draw


def main():
    cap = cv2.VideoCapture(1)
    detector = ActionDetector()
    draw = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame horizontally

        if draw is None:
            draw = Draw(frame.shape)

        processed_frame, action = detector.detect_action(frame)


        output_frame = draw.draw(processed_frame, action,
                                            detector.prev_hand_landmarks,
                                            detector.mp_hands)

        # Display the action on the frame
        if action:
            cv2.putText(output_frame, f"Action: {action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Blackboard AI', output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            draw.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
