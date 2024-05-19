set -e

mkdir -p /output/

# Needs to be run inside triton docker
model-analyzer profile \
    --model-repository /models/ \
    --profile-models ensemble_model \
    --triton-launch-mode remote \
    --triton-docker-image custom_triton:latest \
    --output-model-repository-path /output/profile \
    --export-path profile_results \
   --triton-http-endpoint 0.0.0.0:8000 \
   --triton-grpc-endpoint 0.0.0.0:8001 \
    --override-output-model-repository

#    --triton-metrics-url 0.0.0.0:8002 \
