#!/bin/bash
mkdir -p data/external/mit-place-pulse
curl -L -o data/external/mit-place-pulse/data.zip\
  https://www.kaggle.com/api/v1/datasets/download/shubham6147/mit-place-pulse
unzip data/external/mit-place-pulse/data.zip -d data/external/mit-place-pulse
rm data/external/mit-place-pulse/data.zip