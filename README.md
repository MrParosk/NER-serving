# NER-serving

Project for serving a Hugging-face NER model with Triton inference server. Below is the setup steps:

1. Generate onnx model by running: ```python convert_to_onnx.py```
2. Copy over model to model_repository folder with ```bash scripts/copy_model.sh```
3. Start triton with ```bash scripts/run_triton.sh```
4. Benchmark requests with ```python send_requests.py```
