from frontend.video_stream import VideoStream
from backend.pipeline import process_frame
from frontend.visualization import draw_results
import cv2
import os


def main():
    stream = VideoStream(0)
    while True:
        frame = stream.get_frame()
        if frame is None:
            break
        
        results = process_frame(frame)
        
        output = draw_results(frame,results)
        
        cv2.imshow("Emotion Detection in Crowd",output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    stream.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()         
    