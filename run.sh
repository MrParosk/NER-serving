set -e

docker build . -t custom_triton

docker run --gpus=all --rm --net=host -v ${PWD}/model_repository:/models \
    custom_triton:latest tritonserver --model-repository=/models --model-control-mode=poll
