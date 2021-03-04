FROM python:3.9.1-buster

RUN apt-get update \
&&  apt-get install \
      libgeos-c1v5 \
      libgeos-dev \
      libproj-dev \
      proj-bin -y
RUN python -m pip install numpy \
&&  python -m pip install --no-binary :all: shapely
RUN mkdir -p /code
WORKDIR /code
ADD requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

ADD ./* ./

CMD python -u visualisation.py