/**
 * filters.h
 * Shivang Patel (shivang2402) - 2026-01-23
 * Image filter function declarations.
 */

#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>
#include <vector>

int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int spotlight(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);
int neonEdges(cv::Mat &src, cv::Mat &dst);
int cartoon(cv::Mat &src, cv::Mat &dst, int levels);
int digitalFog(cv::Mat &src, cv::Mat &depthMap, cv::Mat &dst);

#endif
