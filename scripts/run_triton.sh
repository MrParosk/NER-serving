set -e

docker build . -t custom_triton

docker run --gpus=all --rm --net=host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ${PWD}/model_repository:/models \
    -v ${PWD}/scripts:/scripts \
    custom_triton:latest tritonserver \
    --model-repository=/models --model-control-mode=explicit # poll
