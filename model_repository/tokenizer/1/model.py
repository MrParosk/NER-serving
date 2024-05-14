import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        model_name = "dslim/bert-large-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def execute(self, requests):
        responses = []     
        for request in requests:
            in_text = pb_utils.get_input_tensor_by_name(request, "input_text")
            in_text = in_text.as_numpy()
            decoded_text = in_text.astype(str)[:,0].tolist()

            output = self.tokenizer(decoded_text, return_tensors="np", padding=True)
            input_ids = output["input_ids"]
            token_type_ids = output["token_type_ids"]
            attention_mask = output["attention_mask"]

            tensor1 = pb_utils.Tensor("input_ids", input_ids)
            tensor2 = pb_utils.Tensor("token_type_ids", token_type_ids)
            tensor3 = pb_utils.Tensor("attention_mask", attention_mask)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[tensor1, tensor2, tensor3]
            )
            
            responses.append(inference_response)
        return responses
