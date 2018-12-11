#pragma once

#include <string>
#include <vector>

#include <inference_engine.hpp>
#include <opencv2/core/core.hpp>

#include "movidius_ros/human_pose.h"

namespace human_pose_estimation {
class HumanPoseEstimator {
 public:
  HumanPoseEstimator(const std::string &modelPath,
                     const std::string &targetDeviceName,
                     bool enablePerformanceReport = false);
  std::vector<GatePose> estimate(const cv::Mat &image);
  ~HumanPoseEstimator();

 private:
  void preprocess(const cv::Mat &image, float *buffer) const;
  InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat);
  template<typename T>
  void matU8ToBlob(const cv::Mat &orig_image, InferenceEngine::Blob::Ptr &blob);
  std::string type2str(int type);
//  bool inputWidthIsChanged(const cv::Size &imageSize);

  int minJointsNumber;
  int stride;
//  cv::Vec4i pad;
//  cv::Vec3f meanPixel;
//  float minPeaksDistance;
//  float midPointsScoreThreshold;
//  float foundMidPointsRatioThreshold;
//  float minSubsetScore;
  cv::Size inputLayerSize;
//  int upsampleRatio;
  InferenceEngine::InferencePlugin plugin;
  InferenceEngine::CNNNetwork network;
  InferenceEngine::ExecutableNetwork executableNetwork;
  InferenceEngine::InferRequest request;
  InferenceEngine::CNNNetReader netReader;
  std::string pafsBlobName;
//  std::string heatmapsBlobName;
  bool enablePerformanceReport;
  std::string modelPath;
};
}  // namespace human_pose_estimation
