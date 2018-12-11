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

#include <vector>

#include "movidius_ros/human_pose.h"

namespace human_pose_estimation {
GatePose::GatePose() {
  r_ = 0.0;
  theta_ = 0.0;
  phi_ = 0.0;
  yaw_ = 0.0;
  var_r_ = 1.0;
  var_theta_ = 1.0;
  var_phi_ = 1.0;
  var_yaw_ = 1.0;
}
GatePose::GatePose(const float r,
                   const float theta,
                   const float phi,
                   const float yaw,
                   const float var_r,
                   const float var_theta,
                   const float var_phi,
                   const float var_yaw)
    : r_(r),
      theta_(theta),
      phi_(phi),
      yaw_(yaw),
      var_r_(var_r),
      var_theta_(var_theta),
      var_phi_(var_phi),
      var_yaw_(var_yaw) {}
}  // namespace human_pose_estimation
