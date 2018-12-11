/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

namespace human_pose_estimation {
struct GatePose {
  GatePose();
  GatePose(const float r,
           const float theta,
           const float phi,
           const float yaw,
           const float var_r,
           const float var_theta,
           const float var_phi,
           const float var_yaw);

    float r_, theta_, phi_, yaw_, var_r_, var_theta_, var_phi_, var_yaw_;
//    std::vector<cv::Point2f> keypoints;
//    float score;
};
}  // namespace human_pose_estimation
