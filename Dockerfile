# source of image https://github.com/Kaggle/docker-python/blob/master/Dockerfile

FROM kaggle/python

RUN apt-get -y update && \
    apt-get -y install vim \
                       htop \
                       git \
                       wget \
                       sudo \
                       unzip \
                       tmux

COPY requirements.txt /root/requirements.txt
RUN pip install -U pip && pip install -r /root/requirements.txt

