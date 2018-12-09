#!/usr/bin/env bash

# Model optimizer command:
# FP16
python3 mo_tf.py --input_model /home/elia/Desktop/movidius_checkpoint/181203/frozen_graph.pb --model_name optimized_graph_FP16 --output_dir /home/elia/Desktop/movidius_checkpoint/181203/ --data_type FP16 --disable_fusing
# FP32
python3 mo_tf.py --input_model /home/elia/Desktop/movidius_checkpoint/181203/frozen_graph.pb --model_name optimized_graph_FP32 --output_dir /home/elia/Desktop/movidius_checkpoint/181203/ --data_type FP32 --disable_fusing


python3


python train_reactive_learner.py --train_mean=True --resume_train=False --train_dir=../data/ICRA/simulation/Training_mean/ --val_dir=../data/ICRA/simulation/Validation_mean/ --checkpoint_dir=../results/181204_debug --max_epochs=50

python train_network.py --network_topology=simple --train_dir=../data/Training/ --val_dir=../data/Validation/ --max_epochs=50



sleep 5s
# train mean of normal-only network
#python train_reactive_learner.py --train_mean=True --train_dir=../data/IROS18/real_world/Normal/Training_mean/ --val_dir=../data/IROS18/real_world/Normal/Validation_mean/ --checkpoint_dir=../results/181002_normal

sleep 5s
# train variance of normal-only network
python train_reactive_learner.py --train_mean=False --resume_train --train_dir=../data/IROS18/real_world/Normal/Training_variance/ --val_dir=../data/IROS18/real_world/Normal/Validation_variance/ --checkpoint_dir=../results/181002_normal --max_epochs=50

sleep 5s
# train mean of jungle network
#python train_reactive_learner.py --train_mean=True --train_dir=../data/IROS18/real_world/Jungle/Training_mean/ --val_dir=../data/IROS18/real_world/Jungle/Validation_mean/ --checkpoint_dir=../results/181002_jungle

sleep 5s
# train variance of jungle network
python train_reactive_learner.py --train_mean=False --resume_train --train_dir=../data/IROS18/real_world/Jungle/Training_mean/ --val_dir=../data/IROS18/real_world/Jungle/Validation_mean/ --checkpoint_dir=../results/181002_jungle --max_epochs=50

sleep 5s
# train mean of dynamic network
python train_reactive_learner.py --train_mean=True --train_dir=../data/IROS18/real_world/Dynamic/Training_mean/ --val_dir=../data/IROS18/real_world/Dynamic/Validation_mean/ --checkpoint_dir=../results/181002_dynamic

sleep 5s
# train variance of dynamic network
python train_reactive_learner.py --train_mean=False --resume_train --train_dir=../data/IROS18/real_world/Dynamic/Training_mean/ --val_dir=../data/IROS18/real_world/Dynamic/Validation_mean/ --checkpoint_dir=../results/181002_dynamic --max_epochs=50




# Debug
python train_reactive_learner.py --train_mean=True --resume_train=False --train_dir=../data/IROS18/real_world/All/Training_mean/ --val_dir=../data/IROS18/real_world/All/Validation_mean/ --checkpoint_dir=../results/181122_all --max_epochs=50
