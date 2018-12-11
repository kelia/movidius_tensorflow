#include "movidius_ros/movidius_ros.h"

#include "geometry_msgs/PoseWithCovarianceStamped.h"

#include <opencv2/opencv.hpp>

#include <movidius_ros/common.h>

namespace movidius_ros {

using namespace InferenceEngine;

MovidiusRos::MovidiusRos
    (const ros::NodeHandle &nh, const ros::NodeHandle &pnh) :
    nh_(nh), pnh_(pnh) {

  if (!loadParameters()) {
    ROS_ERROR("[%s] Could not load parameters.", pnh_.getNamespace().c_str());
  }
  setupInferenceEngine();
  image_sub_ = nh_.subscribe("image_raw", 1, &MovidiusRos::imageCallback, this);
  prediction_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("cnn_prediction", 0);
}

int MovidiusRos::setupInferenceEngine() {
  try {
    std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

    estimator_.reset(new human_pose_estimation::HumanPoseEstimator(model_, device_, false));

    std::cout << "Instantiated estimator" << std::endl;
    performInference(cv::imread(query_img_, cv::IMREAD_COLOR));
  }
  catch (const std::exception &error) {
    std::cerr << "[ ERROR ] " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
  catch (...) {
    std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "[ INFO ] Execution successful" << std::endl;
  return EXIT_SUCCESS;
}

void MovidiusRos::performInference(cv::Mat image) {
  int index = 0;
  int num_inferences = 1000;

  if (!image.data) {
    ROS_INFO(" No image data \n ");
  }
  std::vector<human_pose_estimation::GatePose> poses;
  ros::WallTime start = ros::WallTime::now();
  while (index < num_inferences) {
    poses = estimator_->estimate(image);

    geometry_msgs::PoseWithCovarianceStamped msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "hummingbird/base_link";

    float r = poses[0].r_;
    float theta = poses[0].theta_;
    float phi = poses[0].phi_;
    float yaw = poses[0].yaw_;

    msg.pose.pose.position.x = radius_normalization_ * r * std::sin(theta) * std::cos(phi);
    msg.pose.pose.position.y = radius_normalization_ * r * std::sin(theta) * std::sin(phi);
    msg.pose.pose.position.z = radius_normalization_ * r * std::cos(theta);
    msg.pose.pose.orientation.w = std::cos(yaw / 2.0);
    msg.pose.pose.orientation.x = 0.0;
    msg.pose.pose.orientation.y = 0.0;
    msg.pose.pose.orientation.z = std::sin(yaw / 2.0);

    prediction_pub_.publish(msg);
//
    index++;
  }
  std::cout << "[" << poses[0].r_ << ", " << poses[0].theta_ << ", " << poses[0].phi_ << ", " << poses[0].yaw_ << ", "
            << poses[0].var_r_ << ", " << poses[0].var_theta_ << ", " << poses[0].var_phi_ << ", " << poses[0].var_yaw_
            << "]" << std::endl;
  ROS_INFO("Computation for 1000 inferences took %f sec", (ros::WallTime::now() - start).toSec());
}

void MovidiusRos::imageCallback(const sensor_msgs::ImageConstPtr &msg) {

  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  performInference(cv_ptr->image);
}

bool MovidiusRos::loadParameters() {
  bool check = true;

  check &= pnh_.getParam("device", device_);
  check &= pnh_.getParam("model", model_);
  check &= pnh_.getParam("query_img", query_img_);
  check &= pnh_.getParam("radius_normalization", radius_normalization_);
  return check;
}

} // namespace movidius_ros
