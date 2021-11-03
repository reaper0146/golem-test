FROM python:3.8.7-slim

RUN pip install --upgrade pip
RUN pip install --upgrade pip wheel
RUN pip install lstm
COPY dataset /golem/dataset
RUN ls -lh /golem/dataset

WORKDIR /golem/work
VOLUME /golem/work /golem/output /golem/resource
