import json
import numpy as np

from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        params = json.loads(args["model_config"])["parameters"]
        model_name = params["model_name"]["string_value"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def execute(self, requests):
        responses = []     
        for request in requests:
            in_text = pb_utils.get_input_tensor_by_name(request, "input_text")
            in_text = in_text.as_numpy()
            decoded_text = in_text.astype(str)[:,0].tolist()

            output = self.tokenizer(decoded_text, return_tensors="np", padding=True, return_offsets_mapping=True)
            input_ids = output["input_ids"].astype(np.int32)
            token_type_ids = output["token_type_ids"].astype(np.int32)
            attention_mask = output["attention_mask"].astype(np.int32)
            offset_mapping = output["offset_mapping"]
            offset_mapping = np.array(offset_mapping).astype(np.int32)

            tensor1 = pb_utils.Tensor("input_ids", input_ids)
            tensor2 = pb_utils.Tensor("token_type_ids", token_type_ids)
            tensor3 = pb_utils.Tensor("attention_mask", attention_mask)
            tensor4 = pb_utils.Tensor("offset_mapping", offset_mapping)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[tensor1, tensor2, tensor3, tensor4]
            )
            
            responses.append(inference_response)
        return responses
