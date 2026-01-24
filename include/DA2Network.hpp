/**
 * DA2Network.hpp
 * Shivang Patel (shivang2402) - 2026-01-23
 * Depth Anything V2 wrapper using ONNX Runtime.
 */

#ifndef DA2NETWORK_HPP
#define DA2NETWORK_HPP

#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class DA2Network {
public:
  DA2Network() : session_(nullptr), env_(nullptr), initialized_(false) {}
  ~DA2Network() {
    session_ = nullptr;
    env_ = nullptr;
  }

  bool init(const std::string &modelPath) {
    try {
      env_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DA2Network");
      Ort::SessionOptions opts;
      opts.SetIntraOpNumThreads(4);
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      session_ = new Ort::Session(*env_, modelPath.c_str(), opts);

      Ort::AllocatorWithDefaultOptions alloc;
      inputName_ = session_->GetInputNameAllocated(0, alloc).get();
      outputName_ = session_->GetOutputNameAllocated(0, alloc).get();
      inputShape_ =
          session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

      initialized_ = true;
      std::cout << "DA2Network initialized successfully" << std::endl;
      std::cout << "Input shape: ";
      for (auto s : inputShape_)
        std::cout << s << " ";
      std::cout << std::endl;
      return true;
    } catch (const Ort::Exception &e) {
      std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
      return false;
    }
  }

  bool isInitialized() const { return initialized_; }

  bool process(cv::Mat &src, cv::Mat &dst) {
    if (!initialized_)
      return false;

    try {
      int inH = (inputShape_[2] > 0) ? inputShape_[2] : 518;
      int inW = (inputShape_[3] > 0) ? inputShape_[3] : 518;

      cv::Mat resized, floatImg;
      cv::resize(src, resized, cv::Size(inW, inH));
      resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
      cv::cvtColor(floatImg, floatImg, cv::COLOR_BGR2RGB);

      // Normalize with ImageNet mean/std
      cv::Scalar mean(0.485, 0.456, 0.406), std(0.229, 0.224, 0.225);
      std::vector<cv::Mat> ch;
      cv::split(floatImg, ch);
      for (int i = 0; i < 3; i++)
        ch[i] = (ch[i] - mean[i]) / std[i];
      cv::merge(ch, floatImg);

      // Convert to NCHW
      std::vector<float> tensor(3 * inH * inW);
      for (int c = 0; c < 3; c++)
        for (int h = 0; h < inH; h++)
          for (int w = 0; w < inW; w++)
            tensor[c * inH * inW + h * inW + w] =
                floatImg.at<cv::Vec3f>(h, w)[c];

      // Run inference
      std::vector<int64_t> dims = {1, 3, inH, inW};
      auto mem =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      auto inTensor = Ort::Value::CreateTensor<float>(
          mem, tensor.data(), tensor.size(), dims.data(), 4);
      const char *inNames[] = {inputName_.c_str()};
      const char *outNames[] = {outputName_.c_str()};
      auto out =
          session_->Run(Ort::RunOptions{}, inNames, &inTensor, 1, outNames, 1);

      // Process output
      auto &outT = out[0];
      auto shape = outT.GetTensorTypeAndShapeInfo().GetShape();
      int outH = shape[shape.size() - 2], outW = shape[shape.size() - 1];
      float *data = outT.GetTensorMutableData<float>();

      cv::Mat depth(outH, outW, CV_32FC1, data);
      double minV, maxV;
      cv::minMaxLoc(depth, &minV, &maxV);
      cv::Mat norm;
      depth.convertTo(norm, CV_8UC1, 255.0 / (maxV - minV),
                      -minV * 255.0 / (maxV - minV));
      cv::resize(norm, dst, src.size());
      return true;
    } catch (...) {
      return false;
    }
  }

private:
  Ort::Session *session_;
  Ort::Env *env_;
  std::string inputName_, outputName_;
  std::vector<int64_t> inputShape_;
  bool initialized_;
};

#endif
