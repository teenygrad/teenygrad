"""Reference fixtures for tests/test_functiongemma.rs: FunctionGemma
(`google/functiongemma-270m-it`) on a tool-calling prompt, in FP64 on the CPU.

The prompt is a user question plus one tool, rendered with the model's own chat
template. The reference runs the prompt as one step, then greedy-decodes `--steps`
tokens one at a time with the KV cache, recording the logits at each step:

    <out>/
        prompt_ids.txt          the prompt's token ids, space-separated
        generated_ids.txt       the greedy continuation, space-separated
        logits_0.bin            the prompt's last-token logits [vocab], f32
        logits_<i>.bin          logits after feeding generated token i-1 (i = 1..steps-1)
        text.txt                the decoded continuation

Usage (needs torch and transformers, and access to the gated model):

    python gen_reference.py --out ~/.cache/teeny-llm/fixtures/functiongemma [--steps 24]
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/functiongemma-270m-it"
MESSAGES = [{"role": "user", "content": "What is the weather in Paris?"}]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=24)
    a = p.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float64).eval()
    ids = tok.apply_chat_template(MESSAGES, tools=TOOLS, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"]

    def write(name, t):
        t.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(out, name))

    generated = []
    with torch.no_grad():
        o = model(ids, use_cache=True)
        past = o.past_key_values
        logits = o.logits[0, -1]
        for i in range(a.steps):
            write(f"logits_{i}.bin", logits)
            nxt = int(torch.argmax(logits))
            generated.append(nxt)
            if i + 1 == a.steps:
                break
            o = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[0, -1]

    with open(os.path.join(out, "prompt_ids.txt"), "w") as f:
        f.write(" ".join(map(str, ids[0].tolist())))
    with open(os.path.join(out, "generated_ids.txt"), "w") as f:
        f.write(" ".join(map(str, generated)))
    with open(os.path.join(out, "text.txt"), "w") as f:
        f.write(tok.decode(generated))
    print(ids.shape[1], "prompt tokens;", repr(tok.decode(generated)))


if __name__ == "__main__":
    main()
