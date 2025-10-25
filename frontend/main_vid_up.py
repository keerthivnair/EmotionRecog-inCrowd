from backend.pipeline import process_frame
from frontend.visualization import draw_results
import cv2

def main():
    cap = cv2.VideoCapture("data/vidCrowd.mp4") 
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process each frame
        results = process_frame(frame)
        output = draw_results(frame.copy(), results)

        cv2.imshow("Live Facial Emotion Recognition", output)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
