#!/usr/bin/env bash
python train_network.py --network_topology=simple --train_dir=../data/Training/ --val_dir=../data/Validation/ --max_epochs=50
