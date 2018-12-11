#!/bin/bash         


# set up simulation, will be done by hand
TRAIN_DIR=${HOME}/learning_data/01_Training/
VAL_DIR=${HOME}/learning_data/02_Validation/
LEARNING_DIR=${HOME}/catkin_ws/src/rpg_drone_racing/learning/realsense_learner/src/rs_learner
TRAIN=false

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

    timeout 1s rostopic pub /hummingbird/set_dagger_constants std_msgs/Float32MultiArray """
    data: [1.0, 
           0.0, 
           0.8, 
           0.5, 
           0.5, 
           -0.5, 
           2.0] 
    """
    # [use_prob_dagger, 
    #  reduce probability of mb-approach, 
    #  max_offset start, 
    #  max_offset end, 
    #  gradient, 
    #  bias,
    #  upper_bound]

    # start network prediction
    echo "Turn on network"
    timeout 1s rostopic pub /hummingbird/state_change std_msgs/Bool 'True'
    
    # start navigation
    #echo "Start navigation"
    #timeout 2s rostopic pub /hummingbird/start_navigation std_msgs/Empty

    # setup environment and start
    timeout 1s rostopic pub /hummingbird/setup_environment std_msgs/Empty

    # data is collected for 60 sec
    echo "Collecting data..."
    sleep 60s

    # stop algorithm
    echo "Stop collecting data"
    timeout 2s rostopic pub /hummingbird/hard_stop std_msgs/Empty

    # stop network prediction
    echo "Turn off network"
    timeout 2s rostopic pub /hummingbird/state_change std_msgs/Bool 'False'

    #### retrain network with data
    if [ "$TRAIN" = true ] 
    then
      if [ $FIRST_ITERATION -gt 0 ] 
      then
          python ${LEARNING_DIR}/train.py --checkpoint_dir=${LEARNING_DIR}/results/171221_sinusoidal_gates --train_dir=${TRAIN_DIR} --val_dir=${VAL_DIR} --max_epochs=20 --summary_freq=100 --resume_train=False
          FIRST_ITERATION=0
      else
          python ${LEARNING_DIR}/train.py --checkpoint_dir=${LEARNING_DIR}/results/171221_sinusoidal_gates --train_dir=${TRAIN_DIR} --val_dir=${VAL_DIR} --max_epochs=20 --summary_freq=100 --resume_train=True
      fi
    fi
    ####

    # increase run index by one
    HIGHEST_RUN_INDEX=$((HIGHEST_RUN_INDEX+1))

done


#python ../train.py --checkpoint_dir=../results/171212_failure_detection --train_dir=/home/eliakaufmann/learning_data/01_Training/ --max_epochs=20 --summary_freq=100 --resume_train=True
