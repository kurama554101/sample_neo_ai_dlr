# Sample Human Detection

## introduction

TBD

## environment

TBD

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