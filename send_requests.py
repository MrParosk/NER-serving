import tritonclient.http as httpclient
import numpy as np
import random
import time
from concurrent.futures import ProcessPoolExecutor


def get_inputs(num_samples):
    choices = [
        "Hello I'm Omar and I live in Zurich.",
        "hello world",
        "Erik/Palma swish",
        "Two and a half men",
        "Breaking bad",
        "numpy",
        "pandas",
        "Me and you",
        "Random string",
        "hellllo"
    ]

    inputs = []
    for _ in range(num_samples):
        inputs.append(random.choice(choices))
    return inputs



num_samples = 500
num_req = 100


def func():
    client = httpclient.InferenceServerClient(url="localhost:8000")
    output1 = httpclient.InferRequestedOutput("valid", binary_data=True)
    output2 = httpclient.InferRequestedOutput("token_array", binary_data=True)
    output3 = httpclient.InferRequestedOutput("class_name_array", binary_data=True)
    output4 = httpclient.InferRequestedOutput("offset_mapping", binary_data=True)

    s = time.time()
    results = []
    for _ in range(num_req):
        inputs = get_inputs(num_samples)
        inputs_np = np.array(inputs).astype(object)[..., np.newaxis]
        inputs_0 = httpclient.InferInput("input_text", inputs_np.shape, datatype="BYTES")
        inputs_0.set_data_from_numpy(inputs_np, binary_data=True)

        r = client.async_infer(model_name="ensemble_model", inputs=[inputs_0], outputs=[output1, output2, output3, output4])
        results.append(r)

    for r in results:
        r.get_result()

    diff = time.time() - s
    return diff


num_proc = 1

res = []
with ProcessPoolExecutor(max_workers=num_proc) as mp:
    for _ in range(num_proc):
        res.append(mp.submit(func))


global_diff = 0
for r in res:
    global_diff += r.result()

latency = round(global_diff / (num_req * num_proc), 3)
throughput = round((num_req * num_samples * num_proc) / global_diff)

print(f"Latency: {latency} s")
print(f"Throughput: {throughput} samples / s")
