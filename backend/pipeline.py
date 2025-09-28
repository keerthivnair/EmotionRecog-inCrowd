from .face_detection import detect_faces
from .emotion_model import analyze_emotion
import numpy as np

def process_frame(frame):
    results = []
    faces = detect_faces(frame)
    if len(faces) == 0:
        return results

    
    face_imgs = [frame[y:y+h, x:x+w].astype(np.float32) for (x, y, w, h) in faces]

    
    for i, (x, y, w, h) in enumerate(faces):
        emotion, score = analyze_emotion(face_imgs[i])  
        results.append({
            "box": (x, y, w, h),
            "emotion": emotion,
            "score": score
        })

    return results
