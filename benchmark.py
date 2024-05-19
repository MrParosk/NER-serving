from time import time

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification

from generate_data import get_inputs
from convert_to_onnx import CustomBertForTokenClassification

model_name = "dslim/bert-base-NER"


def benchmark_torch(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CustomBertForTokenClassification.from_pretrained(model_name, return_dict=False)
    #model = AutoModelForTokenClassification.from_pretrained(model_name, return_dict=False)
    model = model.to("cuda")
    inputs = get_inputs(100)

    pt_time = []
    for _ in range(100):
        s = time()
        ids = tokenizer(inputs, return_tensors="pt", padding=True)
        ids = ids.to("cuda")

        o = model(**ids)
        pt_time.append(time() - s)

    print(np.mean(pt_time))


def benchmark_onnx(model_name):
    onnx_model = ORTModelForTokenClassification.from_pretrained("benchmark_onnx_model")
    onnx_model = onnx_model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = get_inputs(100)

    onnx_time = []
    for _ in range(100):
        s = time()
        ids_np = tokenizer(inputs, return_tensors="pt", padding=True)
        _ = onnx_model(**ids_np)
        onnx_time.append(time() - s)

    print(np.mean(onnx_time))


def benchmark_pipeline(model_name):
    onnx_model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = get_inputs(100)

    nlp = pipeline("ner", model=onnx_model, tokenizer=tokenizer, device="cuda")

    pipe_time = []
    for _ in range(100):
        s = time()
        _ = nlp(inputs)
        pipe_time.append(time() - s)

    print(np.mean(pipe_time))

benchmark_torch(model_name)
benchmark_onnx(model_name)
benchmark_pipeline(model_name)
