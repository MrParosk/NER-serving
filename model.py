
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from optimum.onnxruntime import ORTModelForTokenClassification
import numpy as np
import random
from time import time
from optimum.exporters.onnx.convert import onnx_export_from_model
import torch


from transformers.models.bert.modeling_bert import BertForTokenClassification
from typing import Optional, Union, Tuple

from optimum.exporters.onnx.model_configs import BertOnnxConfig

from typing import Dict
from collections import OrderedDict
from optimum.exporters.tasks import TasksManager

model_name = "dslim/bert-large-NER"


class CustomBertForTokenClassification(BertForTokenClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        o = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            *args,
            **kwargs
        )
        logits = o[0]
        maxes = torch.max(logits, dim=-1, keepdims=True)
        max_value = maxes[0]
        shifted_exp = torch.exp(logits - max_value)
        prob = shifted_exp / shifted_exp.sum(dim=-1, keepdims=True)
        classes = torch.argmax(prob, dim=-1)
        return classes


register_for_onnx = TasksManager.create_register("onnx", overwrite_existing=True)

@register_for_onnx("custom-bert", "token-classification")
class CustomBertOnnxConfig(BertOnnxConfig):
    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        """
        Dict containing the axis definition of the output tensors to provide to the model.

        Returns:
            `Dict[str, Dict[int, str]]`: A mapping of each output name to a mapping of axis position to the axes symbolic name.
        """

        return OrderedDict({"classes": {0: "batch_size", 1: "sequence_length"}})


def convert_to_onnx(model_name):
    #model = AutoModelForTokenClassification.from_pretrained(model_name)
    model = CustomBertForTokenClassification.from_pretrained(model_name, return_dict=False)
    onnx_export_from_model(model, "onnx_model/", optimize="O3", device="cuda",
                           task="token-classification", custom_onnx_configs={"model": CustomBertOnnxConfig(model.config)}
    )


def get_inputs():
    choices = ["Hello I'm Omar and I live in Zürich.", "hello world"]

    inputs = []
    for _ in range(100):
        inputs.append(random.choice(choices))
    return inputs


def benchmark_torch(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CustomBertForTokenClassification.from_pretrained(model_name, return_dict=False)
    #model = AutoModelForTokenClassification.from_pretrained(model_name, return_dict=False)
    model = model.to("cuda")
    #print(type(model))
    inputs = get_inputs()

    pt_time = []
    for _ in range(100):
        s = time()
        ids = tokenizer(inputs, return_tensors="pt", padding=True)
        ids = ids.to("cuda")

        o = model(**ids)
        pt_time.append(time() - s)

    print(np.mean(pt_time))


def benchmark_onnx(model_name):
    onnx_model = ORTModelForTokenClassification.from_pretrained("onnx_model")
    onnx_model = onnx_model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = get_inputs()

    onnx_time = []
    for _ in range(100):
        s = time()
        ids_np = tokenizer(inputs, return_tensors="pt", padding=True)
        _ = onnx_model(**ids_np)
        onnx_time.append(time() - s)

    print(np.mean(onnx_time))


def benchmark_pipeline(model_name):
    onnx_model = ORTModelForTokenClassification.from_pretrained("onnx_model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = get_inputs()

    nlp = pipeline("ner", model=onnx_model, tokenizer=tokenizer, device="cuda")

    pipe_time = []
    for _ in range(100):
        s = time()
        _ = nlp(inputs)
        pipe_time.append(time() - s)

    print(np.mean(pipe_time))

#benchmark_torch(model_name)
convert_to_onnx(model_name)


#benchmark_onnx(model_name)
#benchmark_pipeline(model_name)

#from transformers import pipeline

#tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)  # return_dict=False

#onnx_model = ORTModelForTokenClassification.from_pretrained("onnx_model")

#nlp = pipeline("ner", model=onnx_model, tokenizer=tokenizer, device="cuda")

# example = ["Erik/Anders", "Sven"]

# ids = tokenizer(example, return_tensors="pt", padding=True)

#ner_results = nlp(example)
#print(ner_results)

#print(ner_results)
#import torch

# print(tokenizer.batch_decode(ids["input_ids"]))
# print(tokenizer.convert_ids_to_tokens(ids["input_ids"][0]))
# with torch.no_grad():

#     o = nlp.preprocess(example)
#     #o = tokenizer(example, return_tensors="pt", padding=True)
#     print(list(o))
#     o = nlp.model(*list(o))
#     print(o)
#     o = nlp.postprocess([o])

#     print(o)

#ner_results = nlp(example)
#print(ner_results)

#pipe
