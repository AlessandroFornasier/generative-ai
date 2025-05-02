xhost +
docker run --net=host -it --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume "$(pwd):/workspace" --workdir /workspace pytorch/pytorch:latest
