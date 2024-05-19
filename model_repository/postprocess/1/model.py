import numpy as np
from numba import njit

import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


@njit
def _convert_predictions(decoded_text, classes, offset_mapping, idx_to_class):
    valid = np.ones_like(classes)
    num_samples, seq_max_len = classes.shape

    class_name_array = np.empty(classes.shape, dtype=np.dtype('U10'))
    token_array = np.empty(classes.shape, dtype=np.dtype('U10'))

    for n_idx in range(num_samples):
        for s_idx in range(seq_max_len):
            c = idx_to_class[classes[n_idx][s_idx]]
            start = offset_mapping[n_idx][s_idx][0]
            end = offset_mapping[n_idx][s_idx][1]

            if start == 0 and end == 0 or c == "O":
                valid[n_idx][s_idx] = 0
                token_array[n_idx][s_idx] = ""
                class_name_array[n_idx][s_idx] = ""
            else:
                token = decoded_text[n_idx][start:end]
                token_array[n_idx][s_idx] = token
                class_name_array[n_idx][s_idx] = c

    return valid, token_array, class_name_array


def convert(decoded_text, classes, offset_mapping, idx_to_class):
    valid, token_array, class_name_array = _convert_predictions(decoded_text, classes, offset_mapping, idx_to_class)

    valid = valid.astype(bool)
    token_array = token_array.astype(object)
    class_name_array = class_name_array.astype(object)
    return valid, token_array, class_name_array


class TritonPythonModel:
    def initialize(self, args):
        # model_name = "dslim/bert-large-NER"
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        pass

    def execute(self, requests):
        responses = []

        # (samples, max_len)
        # (valid, token, start, end)
        #

        for request in requests:
            in_text = pb_utils.get_input_tensor_by_name(request, "input_text")
            in_text = in_text.as_numpy()
            decoded_text = in_text.astype(str)[:,0].tolist()

            #idx_to_class = {0: 'O', 1: 'B-MISC', 2: 'I-MISC', 3: 'B-PER', 4: 'I-PER', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-LOC', 8: 'I-LOC'}
            idx_to_class = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

            offset_mapping = pb_utils.get_input_tensor_by_name(request, "offset_mapping").as_numpy()
            classes = pb_utils.get_input_tensor_by_name(request, "classes").as_numpy()

            valid, token_array, class_name_array = convert(decoded_text, classes, offset_mapping, idx_to_class)

            tensor1 = pb_utils.Tensor("valid", valid)
            tensor2 = pb_utils.Tensor("token_array", token_array)
            tensor3 = pb_utils.Tensor("class_name_array", class_name_array)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[tensor1, tensor2, tensor3]
            )
            
            responses.append(inference_response)
        return responses
