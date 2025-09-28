# Emotion Detection in Crowd – Workflow & Structure

## 1. Video Input
- Capture video (live webcam or file) using **OpenCV** in Python.
- Frames pushed into a processing pipeline.

## 2. Face Detection
- Use **OpenCV Haar Cascade / DNN face detector / MTCNN**.
- Each detected face gets cropped.
- **Parallelizable step** → detect faces in different regions of frame / multiple frames in parallel.

## 3. Emotion Detection (ML model)
- Pre-trained models (choices):
  - **FER2013 CNN models**
  - **DeepFace (Python lib, supports VGGFace, Facenet, etc.)**
  - Lightweight CNNs for real-time inference
- Classify emotions: `{happy, sad, angry, surprised, neutral, disgust, fear}`.
- Map predictions to bounding boxes on each face.

## 4. Parallelism (C++ + OpenMP)
- **Why C++ here?**
  - Heavy computations (face detection + model inference loops) can be parallelized efficiently.
  - Use **OpenMP** to run per-face or per-frame tasks concurrently.
  - Expose results to Python via **PyBind11** or **Cython** for integration with OpenCV UI.
- **Parallel opportunities:**
  - Frame-level parallelism (process multiple frames at once → useful if batching allowed).
  - Face-level parallelism (process multiple faces in a frame at once).
  - Model inference parallelism (batch predictions using threads).

## 5. Visualization
- Overlay in live video feed:
  - Bounding box + label (emotion).
  - Total face count.
  - Statistics (e.g., percentage of "happy" faces live).
- Display in real time with **OpenCV**.

## 6. Novel Features (to stand out)
- Heatmap of emotions over time (timeline graph).
- Detect **dominant crowd emotion** (majority class).
- Alert if aggressive/angry emotion percentage is high (crowd safety application).
- Cloud/offline toggle: run model on GPU server vs edge device.
- Efficiency comparison: sequential vs parallel (graphs).

---

# 🔧 Tools & Requirements
- **Languages**: Python (frontend/UI + OpenCV) + C++ (parallel backend).
- **Libraries**:
  - Python: OpenCV, PyBind11, NumPy, TensorFlow/PyTorch (for model).
  - C++: OpenMP, dlib/opencv-cpp.
- **Datasets**: FER2013, RAF-DB, AffectNet.
- **Hardware**: Multicore CPU; optional GPU acceleration.

---

# ⚡ Efficiency Gains from Parallelization
- **Without parallelism:**
  - For each frame: detect faces → process each face sequentially → infer emotion.
  - Latency increases with number of faces.
- **With OpenMP parallelism:**
  - Distribute face inference across multiple CPU cores.
  - Speedup factor ≈ number of cores (minus overhead).
  - Smooth real-time detection for larger crowds (10+ faces per frame).

**Benchmarking ideas:**
- FPS (frames/sec) sequential vs parallel.
- Latency per face detection.

---

# 📌 Constraints & Contingencies
**Constraints:**
- Real-time requirement (latency < 100ms/frame).
- Limited cores → diminishing returns beyond certain threads.
- Accuracy of ML model vs speed tradeoff.

**Contingencies:**
- If real-time fails → batch-process video and show analysis after run.
- If model too heavy → use smaller CNN or MobileNet variant.
- If OpenMP + C++ integration too complex → simulate parallelism in Python with multiprocessing (backup plan).
