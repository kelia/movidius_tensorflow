#include <memory>

#include "rpg_common/main.h"
#include "movidius_ros/movidius_ros.h"

RPG_COMMON_MAIN {
  ros::init(argc, argv, "movidius_ros");

  movidius_ros::MovidiusRos movidius_ros;

  ros::spin();
  return 0;
}
