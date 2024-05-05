
from transformers import AutoTokenizer, AutoModelForTokenClassification

from optimum.onnxruntime import ORTModelForTokenClassification
import numpy as np
import random
from time import time
from optimum.exporters.onnx.convert import onnx_export_from_model


model_name = "dslim/bert-base-NER-uncased"


def convert_to_onnx(model_name):
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    onnx_export_from_model(model, "onnx_model/", optimize="O3", device="cuda")


def get_inputs():
    choices = ["Hello I'm Omar and I live in Zürich.", "hello world"]

    inputs = []
    for _ in range(500):
        inputs.append(random.choice(choices))
    return inputs


def benchmark_torch(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model = model.to("cuda")

    inputs = get_inputs()

    ids = tokenizer(inputs, return_tensors="pt", padding=True)
    ids = ids.to("cuda")

    pt_time = []
    for _ in range(100):
        s = time()
        _ = model(**ids)
        pt_time.append(time() - s)

    print(np.mean(pt_time))


def benchmark_onnx(model_name):
    onnx_model = ORTModelForTokenClassification.from_pretrained("onnx_model")
    onnx_model = onnx_model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = get_inputs()

    ids_np = tokenizer(inputs, return_tensors="pt", padding=True)

    onnx_time = []
    for _ in range(100):
        s = time()
        _ = onnx_model(**ids_np)
        onnx_time.append(time() - s)

    print(np.mean(onnx_time))


benchmark_torch(model_name)
benchmark_onnx(model_name)
