import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        model_name = "dslim/bert-large-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def execute(self, requests):
        responses = []

        # (samples, max_len)
        # (valid, token, start, end)
        #

        for request in requests:
            in_text = pb_utils.get_input_tensor_by_name(request, "input_text")
            in_text = in_text.as_numpy()
            decoded_text = in_text.astype(str)[:,0]

            idx_to_class = {0: 'O', 1: 'B-MISC', 2: 'I-MISC', 3: 'B-PER', 4: 'I-PER', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-LOC', 8: 'I-LOC'}

            offset_mapping = pb_utils.get_input_tensor_by_name(request, "offset_mapping").as_numpy()
            classes = pb_utils.get_input_tensor_by_name(request, "classes").as_numpy()
            
            valid = np.ones_like(classes, dtype=bool)
            num_samples, seq_max_len = classes.shape

            class_name_array = np.empty(classes.shape, dtype=object)
            token_array = np.empty(classes.shape, dtype=object)

            for n_idx in range(num_samples):
                for s_idx in range(seq_max_len):
                    c = idx_to_class[classes[n_idx][s_idx]]
                    start = offset_mapping[n_idx][s_idx][0]
                    end = offset_mapping[n_idx][s_idx][1]

                    if start == 0 and end == 0:
                        valid[n_idx][s_idx] = False
                    elif c == "O":
                        valid[n_idx][s_idx] = False
                    else:
                        token = decoded_text[n_idx][start:end]
                        token_array[n_idx][s_idx] = token
                        class_name_array[n_idx][s_idx] = c

            tensor1 = pb_utils.Tensor("valid", valid)
            tensor2 = pb_utils.Tensor("token_array", token_array)
            tensor3 = pb_utils.Tensor("class_name_array", class_name_array)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[tensor1, tensor2, tensor3]
            )
            
            responses.append(inference_response)
        return responses
