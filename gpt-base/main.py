import json
import tiktoken
import torch
import torch.nn as nn
from model import GPTModel

torch.manual_seed(123)
with open("config.json", "r") as file:
    GPT_CONFIG_124M = json.load(file)

model = GPTModel(GPT_CONFIG_124M)

text = input("Enter text: ")
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(text)
enc_text = torch.tensor(enc_text)
batch_inp = enc_text.unsqueeze(0)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


out_ids = generate_text_simple(
    model=model, idx=batch_inp, max_new_tokens=10, context_size=1024
)

print(tokenizer.decode(out_ids.squeeze().tolist()))
