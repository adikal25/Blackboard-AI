import cv2
from detect_gestures import ActionDetector

def main():
    cap = cv2.VideoCapture(1)
    detector = ActionDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, action = detector.detect_action(frame)

        # Display the action on the frame
        if action:
            cv2.putText(processed_frame, f"Action: {action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Blackboard AI', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()