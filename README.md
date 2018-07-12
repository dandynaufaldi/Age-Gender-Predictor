# Age-Gender-Predictor

Predict age and gender of person from their face. Using dlib for face alignment.
Perform transfer learning and fine-tuning from IMDB-Wiki Dataset with VGG-16, InceptionV3, and Xception architecture using Keras framework on top of Tensorflow

## Requirement
1. Docker CE and nvidia-docker installed
2. Pull docker image from dandynaufaldi/tf-keras-py3.5 
    
    ```docker pull dandynaufaldi/tf-keras-py3.5```
3. Script for run docker with X11 forwarding (GUI apps), you may add `--rm` after `run` so the container will be self-deleted

```
xhost +local:docker
#xhost -local:docker
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

sudo rm -rf /tmp/.docker.xauth
sudo touch /tmp/.docker.xauth

sudo xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
sudo nvidia-docker run -it --env QT_X11_NO_MITSHM=1 --device=/dev/video0 -e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTH:$XAUTH -e XAUTHORITY=$XAUTH  --name nvidia dandynaufaldi/tf-keras-py3.5
```
4. Get the IMDB-Wiki Dataset