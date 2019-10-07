# Sample Object Detection by using SageMaker Neo Runtime

## introduction

sample object detection and face recognition function

## environment

* Docker
    * if you need to play face recognition codes, don't use docker container.

## install

### Create Docker Image

```
$ docker build -t human_detection:0.1 -f docker/DockerFile .
```

If you don't use cache, set "--no-cache=true".

### Run Docker Container

```
$ docker run -v `pwd`/script:/home/development/script -it --name human_detection_container human_detection:0.1 /bin/bash
```

## use object detection scripts

* if you use the object detection scripts, please run these scripts on docker container.

### use sample inference script

```
$ python3 /home/development/script/sample_infer.py
```

## use face recognition scripts

### add face images

Please do the following process to add the face information.

* add the face images into "script/data/face_data"
* add the face images path and face names into "script/data/face_list.csv"
    * ex) write like ""obama.jpg","obama""
 
### use sample face recognition scripts

```
$ cd script
$ streamlit run run_face_recognition.py
```
