# 3D Semantic segmentation of measurements acquired with LiDAR sensors for traffic scenarios 

Here we have my lincense project and it's about 3D semantic segmentation on LiDAR point clouds with applicability in automotive area. The goal was to develop a CNN which can classify the points from a traffic scenario point cloud. Based on RangeNet++ method i developed 2 types of backbones and 6 types of necks in which I used different approaches to create the layers. 

## System dependences 
$ sudo apt-get update

$ sudo apt-get install -yqq build-essential ninja-build \
python3-dev python3-pip apt-utils curl git cmake unzip autoconf autogen \
libtool mlocate zlib1g-dev python3-numpy python3-wheel wget \
software-properties-common openjdk-8-jdk libpng-dev \
libxft-dev ffmpeg python3-pyqt5.qtopengl

$ sudo updatedb

## System library requirements 

scipy==1.6.3 \
numpy==1.22.3 \
torch==1.13.0 \
tensorflow==2.9.1 \
torchvision==0.13.0 \
opencv-python==3.4.11 \
matplotlib==3.5.1 \
yaml==6.0 

## CNN's configuration files
For each version of CNN we have a configuration file in which are weitten informations like: traning parameters, backbone parameters, neck parameters, head parameters and post-processing parameters.

Path: ./train/tasks/semantic/config

## Training 

1. Open terminal 
2. cd ./train/tasks/semantic
3. python ./train.py -d DATASET_PATH -ac ARCH_CONF_PATH -l LOG_PATH -p PRETRAINED_MODEL_PATH

* -p parameter is not necessary

## Inference 

1. Open terminal 
2. cd ./train/tasks/semantic
3. python ./infer.py -d DATASET_PATH -l PREDICTIONS_DIRECTORY_PATH -m PRETRAINED_MODEL_PATH

* all parameters are necessary

## Evaluation

1. Open terminal 
2. cd ./train/tasks/semantic
3. python ./evaluate_iou.py -d DATASET_PATH -p PREDICTIONS_DIRECTORY_PATH --split TRAIN/TEST

* all parameters are necessary

## Visualization

1. Open terminal 
2. cd ./train/tasks/semantic
3. python ./visualize.py -d DATASET_PATH -s SECVENTION_NUMBER -p PREDICTIONS_DIRECTORY_PATH -pcv TRUE/FALSE -c ARCH_CONF_PATH -med TRUE/FALSE -w IMAGE_WIDTH

* -s parameter is used to specify the number of the sequence we want to run 
* -pvc parameter is used to choose if you want to run 3D point cloud visualization or not
* -med parameter is used to choose if you want to apply a median filter over the projected images or not


