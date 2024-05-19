import random

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
