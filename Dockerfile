FROM nvidia/cuda:10.1-cudnn7-runtime

COPY . /api

# install python
RUN apt-get update -y\
 && apt-get install -y python3 python3-pip

# install ML deps
RUN pip3 install tensorflow-gpu keras numpy matplotlib Pillow

# install web deps
RUN pip3 install flask flask_cors

EXPOSE 8889
WORKDIR /api
CMD python3 /api/app.py


