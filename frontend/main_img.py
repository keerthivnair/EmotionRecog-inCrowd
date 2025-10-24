from backend.pipeline import process_frame
from frontend.visualization import draw_results
import cv2
import os

def main():
    img_path = os.path.join("data", "faces.jpeg")
    frame = cv2.imread(img_path)

    if frame is None:
        print("Could not load image:", img_path)
        return
    results = process_frame(frame)
    output = draw_results(frame.copy(), results)
    cv2.imshow("Emotion Detection in Crowd", output)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
