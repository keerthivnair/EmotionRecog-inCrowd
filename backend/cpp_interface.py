import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../cpp/build"))

import face_parallel

def analyze_faces(face_imgs):
    """
    face_imgs: list of float32 NumPy arrays (HxWxC)
    returns: list of predicted emotions
    """
    return face_parallel.analyze_faces_parallel(face_imgs)
