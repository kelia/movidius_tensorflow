#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>
#include <dirent.h>

#include <inference_engine.hpp>
#include <ext_list.hpp>
#include "movidius_ros/human_pose_estimator.h"

#include <Eigen/Eigen>
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

namespace movidius_ros {

class MovidiusRos {
 public:
  MovidiusRos(const ros::NodeHandle &nh, const ros::NodeHandle &pnh);
  MovidiusRos() :
      MovidiusRos(ros::NodeHandle(), ros::NodeHandle("~")) {
  }

 private:

  int setupInferenceEngine();
  void imageCallback(const sensor_msgs::ImageConstPtr &msg);
  void performInference(cv::Mat image);
  bool loadParameters();

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber image_sub_;

  ros::Publisher prediction_pub_;

  std::string device_;
  std::string model_;
  std::string query_img_;
  double radius_normalization_;

  std::unique_ptr<human_pose_estimation::HumanPoseEstimator> estimator_;
  size_t batch_size_;
  std::string first_output_name_;
  InferenceEngine::InputsDataMap input_info_;
  InferenceEngine::ExecutableNetwork executable_network_;

};

} // namespace movidius_ros
