/**
 * faceDetect.cpp
 * Shivang Patel (shivang2402) - 2026-01-23
 * Face detection using Haar cascades.
 */

#include "faceDetect.h"

static cv::CascadeClassifier faceCascade;
static bool cascadeLoaded = false;

static bool loadCascade() {
  if (cascadeLoaded)
    return true;

  std::vector<std::string> paths = {
      "../data/haarcascade_frontalface_default.xml",
      "data/haarcascade_frontalface_default.xml",
      "/opt/homebrew/share/opencv4/haarcascades/"
      "haarcascade_frontalface_default.xml"};

  for (const auto &p : paths) {
    if (faceCascade.load(p)) {
      cascadeLoaded = true;
      return true;
    }
  }
  return false;
}

int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces) {
  faces.clear();
  if (!loadCascade()) {
    std::cerr << "Error: Could not load face cascade" << std::endl;
    return -1;
  }
  faceCascade.detectMultiScale(grey, faces, 1.1, 3, 0, cv::Size(30, 30));
  return 0;
}

int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, cv::Scalar color) {
  for (const auto &face : faces) {
    cv::rectangle(frame, face, color, 2);
  }
  return 0;
}
