from typing import Optional, Dict
from collections import OrderedDict

import torch
from transformers.models.bert.modeling_bert import BertForTokenClassification
from optimum.exporters.onnx.model_configs import BertOnnxConfig
from optimum.exporters.tasks import TasksManager
from optimum.exporters.onnx.convert import onnx_export_from_model
from transformers import AutoModelForTokenClassification


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
        return classes.to(torch.int32)


register_for_onnx = TasksManager.create_register("onnx", overwrite_existing=True)
@register_for_onnx("custom-bert", "token-classification")
class CustomBertOnnxConfig(BertOnnxConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use int32 instead of int64
        self.int_dtype = "int32"

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        """
        Dict containing the axis definition of the output tensors to provide to the model.

        Returns:
            `Dict[str, Dict[int, str]]`: A mapping of each output name to a mapping of axis position to the axes symbolic name.
        """

        return OrderedDict({"classes": {0: "batch_size", 1: "sequence_length"}})


def convert_to_onnx_for_triton(model_name):
    model = CustomBertForTokenClassification.from_pretrained(model_name, return_dict=False)
    onnx_export_from_model(model, "onnx_model/", optimize="O4", device="cuda", monolith=True,
                           task="token-classification", custom_onnx_configs={"model": CustomBertOnnxConfig(model.config)}
    )


def convert_to_onnx_for_benchmark(model_name):
    model = AutoModelForTokenClassification.from_pretrained(model_name, return_dict=False)
    onnx_export_from_model(model, "benchmark_onnx_model/", optimize="O4", device="cuda", monolith=True,
                           task="token-classification"
    )


if __name__ == "__main__":
    model_name = "dslim/bert-base-NER"
    convert_to_onnx_for_triton(model_name)
    convert_to_onnx_for_benchmark(model_name)
