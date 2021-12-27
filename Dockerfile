FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

RUN apt-get update && apt-get install -y python3 python3-pip python python-pip

COPY . /opt/ml/


RUN cd /opt/ml/code

RUN mkdir metrics

RUN mkdir data

RUN cd data

RUN mkdir images

RUN cd ..

WORKDIR /opt/ml/code/

RUN pip3 install -r requirements.txt

ENV VERSION=None

ENV INPUT_DATA=['s3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-6/output/photo-tagging-job6-mk/annotations/consolidated-annotation/consolidation-request','s3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-7/output/photo-tagging-job7-mk/annotations/consolidated-annotation/consolidation-request/','s3://cerebro-labeling-jobs/photo-tagging/photo-tagging-job-8/output/photo-tagging-job-8-v2/annotations/consolidated-annotation/consolidation-request/']

ENTRYPOINT [ "python"]