FROM nvcr.io/nvidia/tritonserver:24.04-py3

RUN pip install transformers==4.40.2 numba==0.59.1 triton-model-analyzer==1.39.0

