from zeroband.model import Transformer, llama_configs
import torch

def test_forward():
    model_args = llama_configs["debugmodel"]
    model = Transformer(model_args)
    tokens = torch.randint(0, model_args.vocab_size, (1, 10))
    output = model(tokens)
    assert output.shape == (1, 10, model_args.vocab_size)