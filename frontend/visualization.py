import cv2

def draw_results(frame, results):
    """
    results = list of dicts:
    [
        {"box": (x, y, w, h), "emotion": "happy", "score": 0.92},
        ...
    ]
    """
    
    results = sorted(results, key=lambda r: r['box'][2]*r['box'][3], reverse=True)

    
    h_factor = frame.shape[0] / 720  
    w_factor = frame.shape[1] / 1280  
    font_scale = max(0.8, 0.6 * min(h_factor, w_factor))
    thickness = max(2, int(2 * min(h_factor, w_factor)))

    for res in results:
        x, y, w, h = res["box"]
        emotion = res["emotion"]
        score = res["score"]

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)

        
        text_y = max(y - 10, 20)
        cv2.putText(frame,
                    f"{emotion} ({score:.2f})",
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), thickness)

    return frame
