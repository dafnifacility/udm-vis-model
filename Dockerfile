FROM python:3.9.1-buster

RUN apt-get update \
&&  apt-get install \
      libgeos-c1v5 \
      libgeos-dev \
      libproj-dev \
      proj-bin -y
RUN python -m pip install numpy
ADD ./* /
RUN python -m pip install -r requirements.txt

CMD python visualisation.py