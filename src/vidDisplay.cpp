/**
 * vidDisplay.cpp
 * Shivang Patel (shivang2402) - 2026-01-23
 * Live video capture with real-time filters.
 * Keys: q=quit, s=save, c/g/h/p/b/x/y/m/l/f/1/2/3/d/4 = filters
 */

#include "DA2Network.hpp"
#include "faceDetect.h"
#include "filters.h"
#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

static DA2Network *depthNetwork = nullptr;
static bool depthNetworkLoaded = false;
static bool depthModelWarned = false;

static bool ensureDepthNetwork() {
  if (depthNetworkLoaded)
    return true;
  if (depthModelWarned)
    return false;

  if (depthNetwork == nullptr) {
    depthNetwork = new DA2Network();
  }

  std::vector<std::string> paths = {"../data/depth_anything_v2_vits.onnx",
                                    "data/depth_anything_v2_vits.onnx"};
  for (const auto &p : paths) {
    if (depthNetwork->init(p)) {
      depthNetworkLoaded = true;
      return true;
    }
  }

  depthModelWarned = true;
  std::cerr << "Warning: No depth model found in data/" << std::endl;
  return false;
}

int main(int argc, char *argv[]) {
  cv::VideoCapture *capdev = new cv::VideoCapture(0);

  // Retry for macOS permission dialog
  if (!capdev->isOpened()) {
    std::cerr << "Retrying camera..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    delete capdev;
    capdev = new cv::VideoCapture(0);
  }

  if (!capdev->isOpened()) {
    std::cerr << "Unable to open camera" << std::endl;
    return -1;
  }

  cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  std::cout << "Camera resolution: " << refS.width << " x " << refS.height
            << std::endl;

  cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
  std::cout << "Keys: q=quit s=save c/g/h/p/b/x/y/m/l/f/1/2/3/d/4=filters"
            << std::endl;

  cv::Mat frame, displayFrame, depthMap, grey, sobelX, sobelY;
  std::vector<cv::Rect> faces;
  int screenshotCounter = 0;
  char mode = 'c';

  for (;;) {
    *capdev >> frame;
    if (frame.empty())
      break;

    switch (mode) {
    case 'c':
      displayFrame = frame.clone();
      break;
    case 'g':
      cv::cvtColor(frame, displayFrame, cv::COLOR_BGR2GRAY);
      cv::cvtColor(displayFrame, displayFrame, cv::COLOR_GRAY2BGR);
      break;
    case 'h':
      greyscale(frame, displayFrame);
      break;
    case 'p':
      sepia(frame, displayFrame);
      break;
    case 'b':
      blur5x5_2(frame, displayFrame);
      break;
    case 'x':
      sobelX3x3(frame, sobelX);
      cv::convertScaleAbs(sobelX, displayFrame);
      break;
    case 'y':
      sobelY3x3(frame, sobelY);
      cv::convertScaleAbs(sobelY, displayFrame);
      break;
    case 'm':
      sobelX3x3(frame, sobelX);
      sobelY3x3(frame, sobelY);
      magnitude(sobelX, sobelY, displayFrame);
      break;
    case 'l':
      blurQuantize(frame, displayFrame, 10);
      break;
    case 'f':
      displayFrame = frame.clone();
      cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
      detectFaces(grey, faces);
      drawBoxes(displayFrame, faces, cv::Scalar(0, 255, 0));
      break;
    case '1':
      cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
      detectFaces(grey, faces);
      spotlight(frame, displayFrame, faces);
      break;
    case '2':
      neonEdges(frame, displayFrame);
      break;
    case '3':
      cartoon(frame, displayFrame, 10);
      break;
    case 'd':
      if (ensureDepthNetwork() && depthNetwork->process(frame, depthMap)) {
        cv::cvtColor(depthMap, displayFrame, cv::COLOR_GRAY2BGR);
      } else {
        displayFrame = frame.clone();
        cv::putText(displayFrame, "Depth model not loaded", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      }
      break;
    case '4':
      if (ensureDepthNetwork() && depthNetwork->process(frame, depthMap)) {
        digitalFog(frame, depthMap, displayFrame);
      } else {
        displayFrame = frame.clone();
        cv::putText(displayFrame, "Depth model not loaded", cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      }
      break;
    default:
      displayFrame = frame.clone();
      break;
    }

    cv::imshow("Video", displayFrame);
    char key = cv::waitKey(10);

    if (key == 'q' || key == 'Q')
      break;
    if (key == 's' || key == 'S') {
      std::string filename = "../data/screenshot_" +
                             std::to_string(std::time(nullptr)) + "_" +
                             std::to_string(screenshotCounter++) + ".png";
      cv::imwrite(filename, displayFrame);
      std::cout << "Saved: " << filename << std::endl;
    } else if (std::string("cghpbxymlf1234d").find(key) != std::string::npos) {
      mode = key;
      std::cout << "Mode: " << mode << std::endl;
    }
  }

  delete capdev;
  cv::destroyAllWindows();
  return 0;
}
