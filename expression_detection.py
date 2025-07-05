import cv2
from fer import FER


def main():
    """Detect and display facial expressions from the webcam."""
    detector = FER(mtcnn=True)
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Could not open webcam")
        return

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to capture image")
                break

            # Detect emotions in the current frame
            results = detector.detect_emotions(frame)
            if results:
                # Use the first detected face
                (x, y, w, h) = results[0]["box"]
                emotions = results[0]["emotions"]
                emotion_label = max(emotions, key=emotions.get)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    emotion_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
