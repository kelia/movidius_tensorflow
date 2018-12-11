#!/bin/bash         



timeout 1s rostopic pub /hummingbird/state_change std_msgs/Bool "data: true" 


sleep 3s

timeout 1s rostopic pub /owl/only_network std_msgs/Bool 'True'

# setup environment and start
timeout 1s rostopic pub /owl/setup_environment std_msgs/Empty

# data is collected for 60 sec
echo "Racing..."

