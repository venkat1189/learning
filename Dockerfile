FROM python:3.7-slim

ENV CONTAINER_HOME=/prod-app

WORKDIR $CONTAINER_HOME

COPY . $CONTAINER_HOME


RUN pip install -r requirements.txt