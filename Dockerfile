FROM pytorch/pytorch:latest

RUN pip install pytorch_transformers
RUN pip install flask
RUN pip install flask-restful
RUN pip install scipy
RUN pip install gunicorn

RUN mkdir /home/api
RUN mkdir /home/api/outputs
WORKDIR /home/api

COPY . /home/api
COPY ./outputs /home/api/outputs
