/**
 * filters.cpp
 * Shivang Patel (shivang2402) - 2026-01-23
 * Image filter implementations.
 */

#include "../include/filters.h"
#include <algorithm>
#include <cmath>

// Greyscale using desaturation: (max + min) / 2
int greyscale(cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), src.type());
  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; j++) {
      uchar maxVal = std::max({srcRow[j][0], srcRow[j][1], srcRow[j][2]});
      uchar minVal = std::min({srcRow[j][0], srcRow[j][1], srcRow[j][2]});
      uchar grey = (maxVal + minVal) / 2;
      dstRow[j] = cv::Vec3b(grey, grey, grey);
    }
  }
  return 0;
}

// Sepia tone transformation
int sepia(cv::Mat &src, cv::Mat &dst) {
  dst.create(src.size(), src.type());
  // Calculate center of image
  int rows = src.rows;
  int cols = src.cols;

  for (int i = 0; i < rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < cols; j++) {
      float b = srcRow[j][0], g = srcRow[j][1], r = srcRow[j][2];

      // Calculate vignetting (darken as we get further from center)
      // distance from center normalized to [0, 1] range roughly
      // (1 - dist) used as scaling factor
      // Simple approximation:
      // 1.0 at center, fading to ~0.4 at corners

      // Sepia transform
      float db = 0.272f * r + 0.534f * g + 0.131f * b;
      float dg = 0.349f * r + 0.686f * g + 0.168f * b;
      float dr = 0.393f * r + 0.769f * g + 0.189f * b;

      // Apply vignette
      // Using a simpler cosine-based or distance-based falloff
      // Or just a simple manual falloff to match report claims
      // Let's implement a quick nice vignette:
      double dx = (j - cols / 2.0) / (cols / 2.0); // -1 to 1
      double dy = (i - rows / 2.0) / (rows / 2.0); // -1 to 1
      double distSq = dx * dx + dy * dy;
      double vignette = 1.0 - (distSq * 0.3); // 0.3 strength
      if (vignette < 0)
        vignette = 0;

      dstRow[j][0] = cv::saturate_cast<uchar>(db * vignette);
      dstRow[j][1] = cv::saturate_cast<uchar>(dg * vignette);
      dstRow[j][2] = cv::saturate_cast<uchar>(dr * vignette);
    }
  }
  return 0;
}

// Naive 5x5 blur using at() - for timing comparison
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
  src.copyTo(dst);
  int kernel[5][5] = {{1, 2, 4, 2, 1},
                      {2, 4, 8, 4, 2},
                      {4, 8, 16, 8, 4},
                      {2, 4, 8, 4, 2},
                      {1, 2, 4, 2, 1}};

  for (int i = 2; i < src.rows - 2; i++) {
    for (int j = 2; j < src.cols - 2; j++) {
      int sumB = 0, sumG = 0, sumR = 0;
      for (int ki = -2; ki <= 2; ki++) {
        for (int kj = -2; kj <= 2; kj++) {
          cv::Vec3b px = src.at<cv::Vec3b>(i + ki, j + kj);
          int w = kernel[ki + 2][kj + 2];
          sumB += px[0] * w;
          sumG += px[1] * w;
          sumR += px[2] * w;
        }
      }
      dst.at<cv::Vec3b>(i, j) = cv::Vec3b(sumB / 100, sumG / 100, sumR / 100);
    }
  }
  return 0;
}

// Optimized separable 5x5 blur using row pointers
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
  cv::Mat temp;
  src.copyTo(temp);
  src.copyTo(dst);
  int k[5] = {1, 2, 4, 2, 1};

  // Horizontal pass
  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *tempRow = temp.ptr<cv::Vec3b>(i);
    for (int j = 2; j < src.cols - 2; j++) {
      int sB = 0, sG = 0, sR = 0;
      for (int x = -2; x <= 2; x++) {
        sB += srcRow[j + x][0] * k[x + 2];
        sG += srcRow[j + x][1] * k[x + 2];
        sR += srcRow[j + x][2] * k[x + 2];
      }
      tempRow[j] = cv::Vec3b(sB / 10, sG / 10, sR / 10);
    }
  }

  // Vertical pass
  for (int i = 2; i < src.rows - 2; i++) {
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; j++) {
      int sB = 0, sG = 0, sR = 0;
      for (int y = -2; y <= 2; y++) {
        cv::Vec3b *tRow = temp.ptr<cv::Vec3b>(i + y);
        sB += tRow[j][0] * k[y + 2];
        sG += tRow[j][1] * k[y + 2];
        sR += tRow[j][2] * k[y + 2];
      }
      dstRow[j] = cv::Vec3b(sB / 10, sG / 10, sR / 10);
    }
  }
  return 0;
}

// Sobel X (positive right): [-1 0 1] * [1 2 1]^T
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
  cv::Mat temp(src.size(), CV_16SC3, cv::Scalar(0));
  dst.create(src.size(), CV_16SC3);
  dst.setTo(0);

  int hK[3] = {-1, 0, 1}, vK[3] = {1, 2, 1};

  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);
    for (int j = 1; j < src.cols - 1; j++) {
      for (int c = 0; c < 3; c++) {
        tempRow[j][c] = srcRow[j - 1][c] * hK[0] + srcRow[j][c] * hK[1] +
                        srcRow[j + 1][c] * hK[2];
      }
    }
  }

  for (int i = 1; i < src.rows - 1; i++) {
    cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);
    for (int j = 0; j < src.cols; j++) {
      for (int c = 0; c < 3; c++) {
        dstRow[j][c] = temp.ptr<cv::Vec3s>(i - 1)[j][c] * vK[0] +
                       temp.ptr<cv::Vec3s>(i)[j][c] * vK[1] +
                       temp.ptr<cv::Vec3s>(i + 1)[j][c] * vK[2];
      }
    }
  }
  return 0;
}

// Sobel Y (positive up): [1 2 1] * [1 0 -1]^T
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
  cv::Mat temp(src.size(), CV_16SC3, cv::Scalar(0));
  dst.create(src.size(), CV_16SC3);
  dst.setTo(0);

  int hK[3] = {1, 2, 1}, vK[3] = {1, 0, -1};

  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3s *tempRow = temp.ptr<cv::Vec3s>(i);
    for (int j = 1; j < src.cols - 1; j++) {
      for (int c = 0; c < 3; c++) {
        tempRow[j][c] = srcRow[j - 1][c] * hK[0] + srcRow[j][c] * hK[1] +
                        srcRow[j + 1][c] * hK[2];
      }
    }
  }

  for (int i = 1; i < src.rows - 1; i++) {
    cv::Vec3s *dstRow = dst.ptr<cv::Vec3s>(i);
    for (int j = 0; j < src.cols; j++) {
      for (int c = 0; c < 3; c++) {
        dstRow[j][c] = temp.ptr<cv::Vec3s>(i - 1)[j][c] * vK[0] +
                       temp.ptr<cv::Vec3s>(i)[j][c] * vK[1] +
                       temp.ptr<cv::Vec3s>(i + 1)[j][c] * vK[2];
      }
    }
  }
  return 0;
}

// Gradient magnitude: sqrt(sx^2 + sy^2)
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
  dst.create(sx.size(), CV_8UC3);
  for (int i = 0; i < sx.rows; i++) {
    cv::Vec3s *sxRow = sx.ptr<cv::Vec3s>(i);
    cv::Vec3s *syRow = sy.ptr<cv::Vec3s>(i);
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < sx.cols; j++) {
      for (int c = 0; c < 3; c++) {
        float gx = sxRow[j][c], gy = syRow[j][c];
        dstRow[j][c] = cv::saturate_cast<uchar>(std::sqrt(gx * gx + gy * gy));
      }
    }
  }
  return 0;
}

// Blur then quantize into N levels
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
  blur5x5_2(src, dst);
  int bucket = 255 / levels;
  for (int i = 0; i < dst.rows; i++) {
    cv::Vec3b *row = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < dst.cols; j++) {
      for (int c = 0; c < 3; c++) {
        row[j][c] = (row[j][c] / bucket) * bucket;
      }
    }
  }
  return 0;
}

// Spotlight: greyscale except for face regions
int spotlight(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces) {
  greyscale(src, dst);
  for (const auto &face : faces) {
    int x = std::max(0, face.x), y = std::max(0, face.y);
    int w = std::min(face.width, src.cols - x);
    int h = std::min(face.height, src.rows - y);
    for (int i = y; i < y + h; i++) {
      for (int j = x; j < x + w; j++) {
        dst.ptr<cv::Vec3b>(i)[j] = src.ptr<cv::Vec3b>(i)[j];
      }
    }
  }
  return 0;
}

// Neon edges: bright edges on dark background
int neonEdges(cv::Mat &src, cv::Mat &dst) {
  cv::Mat sobelX, sobelY, mag;
  sobelX3x3(src, sobelX);
  sobelY3x3(src, sobelY);
  magnitude(sobelX, sobelY, mag);

  dst.create(src.size(), src.type());
  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    cv::Vec3b *magRow = mag.ptr<cv::Vec3b>(i);
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; j++) {
      int edge = (magRow[j][0] + magRow[j][1] + magRow[j][2]) / 3;
      if (edge > 30) {
        dstRow[j][0] = cv::saturate_cast<uchar>(srcRow[j][0] * 0.3 + 180);
        dstRow[j][1] = cv::saturate_cast<uchar>(srcRow[j][1] * 0.5 + 200);
        dstRow[j][2] = cv::saturate_cast<uchar>(srcRow[j][2] * 0.3 + 50);
      } else {
        dstRow[j] =
            cv::Vec3b(srcRow[j][0] / 8, srcRow[j][1] / 8, srcRow[j][2] / 8);
      }
    }
  }
  return 0;
}

// Cartoon: quantized colors with black edge outlines
int cartoon(cv::Mat &src, cv::Mat &dst, int levels) {
  cv::Mat quantized, sobelX, sobelY, mag;
  blurQuantize(src, quantized, levels);
  sobelX3x3(src, sobelX);
  sobelY3x3(src, sobelY);
  magnitude(sobelX, sobelY, mag);

  dst.create(src.size(), src.type());
  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *qRow = quantized.ptr<cv::Vec3b>(i);
    cv::Vec3b *mRow = mag.ptr<cv::Vec3b>(i);
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; j++) {
      int edge = (mRow[j][0] + mRow[j][1] + mRow[j][2]) / 3;
      dstRow[j] = (edge > 40) ? cv::Vec3b(0, 0, 0) : qRow[j];
    }
  }
  return 0;
}

// Digital fog: exponential fog based on depth
int digitalFog(cv::Mat &src, cv::Mat &depthMap, cv::Mat &dst) {
  dst.create(src.size(), src.type());
  for (int i = 0; i < src.rows; i++) {
    cv::Vec3b *srcRow = src.ptr<cv::Vec3b>(i);
    uchar *depthRow = depthMap.ptr<uchar>(i);
    cv::Vec3b *dstRow = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < src.cols; j++) {
      float fog = 1.0f - std::exp(-depthRow[j] / 255.0f * 3.0f);
      for (int c = 0; c < 3; c++) {
        dstRow[j][c] =
            cv::saturate_cast<uchar>(srcRow[j][c] * (1 - fog) + 255 * fog);
      }
    }
  }
  return 0;
}
