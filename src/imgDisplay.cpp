/**
 * imgDisplay.cpp
 * Shivang Patel (shivang2402) - 2026-01-23
 * Reads and displays an image. Press 'q' to quit.
 */

#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return -1;
  }

  cv::Mat image = cv::imread(argv[1]);
  if (image.empty()) {
    std::cerr << "Error: Could not open image: " << argv[1] << std::endl;
    return -1;
  }

  cv::namedWindow("Image Display", cv::WINDOW_AUTOSIZE);
  cv::imshow("Image Display", image);
  std::cout << "Press 'q' to quit." << std::endl;

  while (true) {
    if (cv::waitKey(0) == 'q')
      break;
  }

  cv::destroyAllWindows();
  return 0;
}
