"""Lesson 9: continued pretraining of a small causal language model on verbalized baskets (PyTorch, CPU).

"Continued pretraining" is the plain next-token objective the model was first trained with, run a little longer on text
from our own domain. No labels, no task: the model reads baskets and learns to predict every token of them. Every
basket is shown in a fresh random order each time (see Lesson 7). The loss, the mask and the step are written by hand.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lesson7"))
from verbalize import verbalize_basket  # noqa: E402


def encode(tokenizer, baskets, seeds):
    "Each basket as token ids, in the order its seed gives, followed by the end-of-text token."
    return [list(tokenizer.encode(verbalize_basket(b, seed=int(s)))) + [tokenizer.eos_token_id]
            for b, s in zip(baskets, seeds)]


def pad(examples):
    "A batch as token ids padded with 0, and a 0/1 mask that is 1 on real tokens."
    width = max(len(e) for e in examples)
    ids = torch.zeros(len(examples), width, dtype=torch.long)
    mask = torch.zeros(len(examples), width)
    for row, e in enumerate(examples):
        ids[row, :len(e)] = torch.tensor(e)
        mask[row, :len(e)] = 1
    return ids, mask


def next_token_loss(model, ids, mask):
    "Average cross-entropy of predicting every token from the ones before it, ignoring padding."
    logits = model(input_ids=ids[:, :-1], attention_mask=mask[:, :-1].long()).logits.float()
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="none")
    keep = mask[:, 1:].reshape(-1)
    return (loss * keep).sum() / keep.sum()


@torch.no_grad()
def validation_loss(model, tokenizer, baskets, batch_size=16, seed=123):
    "The loss on baskets, each written once in a fixed order. Lower means it predicts held-out baskets better."
    model.eval()
    examples = encode(tokenizer, baskets, np.random.default_rng(seed).integers(2**31, size=len(baskets)))
    total = count = 0.0
    for start in range(0, len(examples), batch_size):
        ids, mask = pad(examples[start:start + batch_size])
        n = float(mask[:, 1:].sum())
        total += float(next_token_loss(model, ids, mask)) * n
        count += n
    model.train()
    return total / count


def pretrain(model_name, baskets, out_dir, val_baskets=None, steps=400, batch_size=16, learning_rate=1e-4, warmup=20,
             eval_every=100, seed=0, log=print):
    """Continue training model_name on baskets for `steps` updates. Saves the model and tokenizer to out_dir and returns
    the history: a list of {step, loss, val_loss, tokens, seconds}. The learning rate warms up linearly, then falls
    along a cosine to a tenth of its peak. Training runs in float32 (half precision loses the small updates)."""
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    bucket = batch_size * 10

    def batches():
        "Endless batches: shuffle, take a bucket, sort it by length so batches waste little padding."
        while True:
            order = rng.permutation(len(baskets))
            for start in range(0, len(order) - bucket + 1, bucket):
                chosen = order[start:start + bucket]
                examples = encode(tokenizer, [baskets[i] for i in chosen], rng.integers(2**31, size=len(chosen)))
                by_length = sorted(range(len(examples)), key=lambda i: len(examples[i]))
                groups = [by_length[i:i + batch_size] for i in range(0, len(by_length), batch_size)]
                for g in rng.permutation(len(groups)):
                    yield pad([examples[i] for i in groups[g]])

    def rate(step):
        if step < warmup:
            return learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    history, started, tokens, recent = [], time.time(), 0, []
    if val_baskets is not None:
        history.append({"step": 0, "loss": None, "val_loss": validation_loss(model, tokenizer, val_baskets), "tokens": 0, "seconds": 0.0})
        log(f"step 0    val loss {history[-1]['val_loss']:.3f}")
    for step, (ids, mask) in zip(range(1, steps + 1), batches()):
        for group in optimizer.param_groups:
            group["lr"] = rate(step - 1)
        loss = next_token_loss(model, ids, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += int(mask.sum())
        recent.append(float(loss))
        if step % eval_every == 0 or step == steps:
            val = validation_loss(model, tokenizer, val_baskets) if val_baskets is not None else None
            history.append({"step": step, "loss": float(np.mean(recent)), "val_loss": val, "tokens": tokens,
                            "seconds": time.time() - started})
            recent = []
            log(f"step {step:<4d} train loss {history[-1]['loss']:.3f}" + (f"  val loss {val:.3f}" if val is not None else "")
                + f"  {tokens / (time.time() - started):,.0f} tokens/s")
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return history
