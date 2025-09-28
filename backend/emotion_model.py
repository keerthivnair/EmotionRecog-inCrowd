from deepface import DeepFace

def analyze_emotion(face_img):
    try:
        analysis = DeepFace.analyze(
            face_img,
            actions=['emotion'],
            enforce_detection=False
        )
        print("Raw analysis:", analysis)  # Debugging

        # Handle both dict and list of dicts
        if isinstance(analysis, list):
            analysis = analysis[0]

        emotion = analysis['dominant_emotion']
        score = analysis['emotion'][emotion]

        return emotion, score

    except Exception as e:
        print("Emotion analysis error:", e)
        return "unknown", 0.0
