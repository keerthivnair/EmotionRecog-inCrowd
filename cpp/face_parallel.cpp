#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>      // needed for std::vector<std::string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <omp.h>
#include <vector>
#include <string>

namespace py = pybind11;

// Absolute path to your ONNX model
const std::string MODEL_PATH = "/home/ubuntu/facialEmotionsInCrowd/cpp/emotions.onnx";

// Load ONNX model globally
cv::dnn::Net net = cv::dnn::readNetFromONNX(MODEL_PATH);

// Emotion labels
const std::vector<std::string> emotions = {"angry","disgust","fear","happy","sad","surprise","neutral"};

// Preprocess face image
cv::Mat preprocess_face(const cv::Mat &face) {
    cv::Mat blob;
    cv::Mat resized;
    cv::resize(face, resized, cv::Size(48, 48));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);
    resized.convertTo(resized, CV_32F, 1.0/255.0);
    cv::dnn::blobFromImage(resized, blob);  // creates 4D blob
    return blob;
}

// Predict emotion for a single face
std::string predict_emotion(const cv::Mat &face) {
    cv::Mat blob = preprocess_face(face);
    net.setInput(blob);
    cv::Mat output = net.forward();

    // Get the class with maximum confidence
    cv::Point classIdPoint;
    double confidence;
    minMaxLoc(output, nullptr, &confidence, nullptr, &classIdPoint);
    int classId = classIdPoint.x;

    return emotions[classId];
}

// Parallel prediction for multiple faces
std::vector<std::string> analyze_faces_parallel(py::list face_list) {
    int n = face_list.size();
    std::vector<std::string> results(n);

    #pragma omp parallel for
    for(int i = 0; i < n; ++i){
        py::array_t<float> face_array = face_list[i].cast<py::array_t<float>>();
        py::buffer_info info = face_array.request();
        int h = info.shape[0];
        int w = info.shape[1];
        int c = info.shape[2];
        float* ptr = static_cast<float*>(info.ptr);

        cv::Mat face(h, w, CV_32FC3, ptr);  
        results[i] = predict_emotion(face);
    }
    return results;
}

// PyBind11 module
PYBIND11_MODULE(face_parallel, m) {
    m.doc() = "Parallel face emotion detection using OpenCV DNN + OpenMP";
    m.def("analyze_faces_parallel", &analyze_faces_parallel, "Analyze multiple faces in parallel");
}
