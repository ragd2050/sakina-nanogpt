# -*- coding: utf-8 -*-
"""
Trains the Sakina comfort-generation model (v10).

Same small architecture as the project's earlier v9 checkpoint (4 layers,
4 heads, 192 dim, 256 context) -- deliberately kept small because it's
trained from scratch on ~340k characters on CPU. This is not a "bigger
model" upgrade; it's the same-scale model retrained on the expanded
11-emotion corpus with more diverse comfort targets per emotion, which is
what was actually broken (repetition / heavy fallback rate) rather than
raw capacity.
"""
import time
import math
import pickle
import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nano_gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
OUT_CKPT = CKPT_DIR / "sakina_v12.pt"
RESUME_CKPT = CKPT_DIR / "sakina_v12_resume.pt"

DEVICE = "cpu"
torch.manual_seed(1337)

# ---- architecture (same scale as v9, block_size trimmed for CPU speed) ----
n_layer, n_head, n_embd, block_size, dropout = 4, 4, 192, 128, 0.15

# ---- optimization ----
batch_size = 8
max_iters = 2200         # total training budget (v12: more data + more comfort-line diversity)
eval_interval = 150
eval_iters = 12
learning_rate = 5e-4
min_lr = 5e-5
warmup_iters = 80
lr_decay_iters = max_iters


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def load_data():
    with open(CKPT_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    vocab_size = meta["vocab_size"]

    def encode(text):
        space_id = stoi.get(" ", 0)
        return [stoi.get(c, space_id) for c in text]

    train_text = (CKPT_DIR / "train_corpus.txt").read_text(encoding="utf-8")
    val_text = (CKPT_DIR / "val_corpus.txt").read_text(encoding="utf-8")
    train_ids = torch.tensor(encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(encode(val_text), dtype=torch.long)
    return train_ids, val_ids, vocab_size


def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=200, help="iterations to run in this invocation")
    args = parser.parse_args()

    train_data, val_data, vocab_size = load_data()

    config = GPTConfig(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, bias=False, vocab_size=vocab_size, dropout=dropout,
    )
    model = GPT(config).to(DEVICE)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, betas=(0.9, 0.99), device_type=DEVICE)

    start_iter = 0
    history = {"iter": [], "train_loss": [], "val_loss": []}
    hist_path = ROOT / "evaluation" / "training_history.pkl"

    if RESUME_CKPT.exists():
        state = torch.load(RESUME_CKPT, map_location=DEVICE)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter_num"]
        if hist_path.exists():
            with open(hist_path, "rb") as f:
                history = pickle.load(f)
        print(f"Resumed from iter {start_iter}")
    else:
        print(f"vocab_size={vocab_size}  train_tokens={len(train_data):,}  val_tokens={len(val_data):,}")

    end_iter = min(start_iter + args.chunk, max_iters)
    t0 = time.time()

    for it in range(start_iter, end_iter + 1):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if it % eval_interval == 0 or it == max_iters:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.0f}s")
            history["iter"].append(it)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])

        if it == end_iter:
            break

        x, y = get_batch(train_data, batch_size, block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model_args = {
        "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd,
        "block_size": block_size, "bias": False, "vocab_size": vocab_size, "dropout": dropout,
    }

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": end_iter,
    }, RESUME_CKPT)

    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    if end_iter >= max_iters:
        torch.save({
            "model": model.state_dict(),
            "model_args": model_args,
            "iter_num": end_iter,
            "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        }, OUT_CKPT)
        print(f"DONE. Final checkpoint -> {OUT_CKPT}")
    else:
        print(f"Chunk done at iter {end_iter}/{max_iters}. Run again to continue.")


if __name__ == "__main__":
    main()
