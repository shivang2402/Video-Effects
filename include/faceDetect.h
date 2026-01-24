/**
 * faceDetect.h
 * Shivang Patel (shivang2402) - 2026-01-23
 * Face detection using Haar cascades.
 */

#ifndef FACEDETECT_H
#define FACEDETECT_H

#include <opencv2/opencv.hpp>

int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces);
int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, cv::Scalar color);

#endif
