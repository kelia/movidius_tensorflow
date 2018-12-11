#!/bin/bash         
TRAIN_DIR=${HOME}/learning_data/Lap_1/
VAL_DIR=${HOME}/learning_data/Lap_2/

# get current run index
for entry in ${TRAIN_DIR}*
do
  HIGHEST_RUN_INDEX=${entry##*_}
done

re='^[0-9]+$'
if ! [[ $HIGHEST_RUN_INDEX =~ $re ]] 
then
   echo "Setting run index to 0"
   HIGHEST_RUN_INDEX=-1
   FIRST_ITERATION=1
else
  HIGHEST_RUN_INDEX=$((ls ${TRAIN_DIR} | tail -1) | tail -c 5) #get largest index
  HIGHEST_RUN_INDEX=${HIGHEST_RUN_INDEX#"${HIGHEST_RUN_INDEX%%[!0]*}"} #remove leading zeros
fi
HIGHEST_RUN_INDEX=$((HIGHEST_RUN_INDEX+1))
echo "Setting run index to $HIGHEST_RUN_INDEX " 

while true; do
    #################################
    # FIRST LAP
    #################################
    # set current run index
    timeout 1s rostopic pub /hummingbird/run_idx std_msgs/Int16 "$HIGHEST_RUN_INDEX"

    echo "Replace quadrotor"
    timeout 1s rostopic pub /hummingbird/copilot/off std_msgs/Empty
    timeout 1s rostopic pub /hummingbird/replace_quad std_msgs/Empty
    sleep 1s
    
    # start quadrotor
    echo "Start quadrotor"
    timeout 1s rostopic pub /hummingbird/copilot/start std_msgs/Empty
    sleep 6s

    timeout 1s rostopic pub /hummingbird/only_network std_msgs/Bool 'False'

    # setup environment and start
    timeout 1s rostopic pub /hummingbird/setup_environment std_msgs/Empty

    # data is collected for 60 sec
    echo "Collecting data..."
    sleep 20s

    # stop algorithm
    echo "Stop collecting data"
    timeout 1s rostopic pub /hummingbird/hard_stop std_msgs/Empty

    ###############################
    # increase run index by one
    ###############################
    HIGHEST_RUN_INDEX=$((HIGHEST_RUN_INDEX+1))
done
