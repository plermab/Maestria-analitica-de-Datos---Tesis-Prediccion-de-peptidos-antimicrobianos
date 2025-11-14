import sys
import os
import argparse
import numpy as np
import re
from datetime import datetime
import math
import pandas as pd
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from tqdm import tqdm
import logging

# Logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
log_file = f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Protein_dataset(Dataset):
    def __init__(self, lines: list[str], tokenizer: Tokenizer):
        self.lines = lines
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = self.tokenizer.encode(line)
        return torch.tensor(line.ids)

def load_data(file: str) -> tuple[list[str], list[str]]:
    lines = []
    prefixes = set()
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            prefix = re.match(r"<\|.*\|>", line).group(0)
            prefixes.add(prefix)
            lines.append(line)
    prefixes = sorted(list(prefixes))
    return lines, prefixes

def init_new_embeddings(model, prefixes):
    if len(prefixes) <= 2:
        logger.info("No new embeddings to initialize.")
        return
    new_embs = torch.zeros((len(prefixes) - 2, model.config.embed_dim)).to(model.device)
    unk_token_emb = model.transformer.wte.weight[-1].detach()
    mean_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.mean()
    std_unk_emb = torch.zeros_like(new_embs) + unk_token_emb.std()
    torch.normal(mean_unk_emb, std_unk_emb, out=new_embs)
    new_embs = torch.cat([model.transformer.wte.weight, new_embs], dim=0)
    model.transformer.wte.weight = torch.nn.Parameter(new_embs, requires_grad=True)
    model.config.vocab_size_emb = new_embs.shape[0]

def get_lr_schedule(optimizer, args, train_steps):
    if args.decay == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, train_steps, num_cycles=0.3 )
    elif args.decay == "linear":
        return get_linear_schedule_with_warmup(optimizer, args.warmup_steps, train_steps)
    elif args.decay == "constant":
        return get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    elif args.decay == "plateau":
        # Baja el LR cuando la eval loss deje de mejorar
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
            cooldown=0, min_lr=1e-6, verbose=True
        )
    else:
        raise ValueError("Invalid learning rate decay type.")

def train_epoch(model, dataset, optimizer, scheduler, epoch, args, scaler):
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0
    pbar = tqdm(total=len(dataloader) // args.accumulation_steps)
    for i, batch in enumerate(dataloader):
        batch = batch.to(model.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(batch, labels=batch).loss / args.accumulation_steps
        scaler.scale(loss).backward()
        if args.debug:  
            grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            logger.debug(f"Gradients - Avg: {np.mean(grads):.3f} | Max: {np.max(grads):.3f}")
        total_loss += loss.item()
        if (i + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update()
    pbar.close()
    logger.info(f"TRAIN epoch {epoch}: loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataset, args, before_train=False):
    model.eval()
    total_loss = 0
    dataloader = DataLoader(dataset, batch_size=1 if before_train else args.batch_size * 4, shuffle=True)
    total_length = len(dataloader)
    pbar = tqdm(total=total_length)
    for batch in dataloader:
        if before_train:
            batch = batch[:, :(batch != 0).sum().item()]
        batch = batch.to(model.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(batch, labels=batch).loss
        total_loss += loss.item()
        pbar.update()
    pbar.close()
    logger.info(f"EVAL loss: {total_loss / total_length}")
    return total_loss / total_length

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        logger.warning("No Slurm job ID found.")

    tokenizer = Tokenizer.from_pretrained(args.model)
    tokenizer.enable_padding(direction="right", pad_id=0, pad_token="<|pad|>", length=100)
    tokenizer.enable_truncation(max_length=args.max_length)

    train_data, prefixes = load_data(args.train_file)
    test_data, prefixes_test = load_data(args.test_file)
    assert prefixes == prefixes_test
    tokenizer.add_tokens(prefixes)

    train_dataset = Protein_dataset(train_data, tokenizer)
    test_dataset = Protein_dataset(test_data, tokenizer)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device)
    if hasattr(model.config, 'hidden_dropout_prob'):
        model.config.hidden_dropout_prob = 0.3 
    if hasattr(model.config, 'attention_probs_dropout_prob'):
        model.config.attention_probs_dropout_prob = 0.3 
    init_new_embeddings(model, prefixes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=False)
    train_steps = args.epochs * len(train_dataset) // (args.batch_size * args.accumulation_steps)
    scheduler = get_lr_schedule(optimizer, args, train_steps)

    if args.eval_before_train:
        logger.info("Runnning evaluation on test set before training...")
        evaluate(model, test_dataset, args, before_train=True)

    train_losses = []
    eval_losses = []
    best_eval_loss = float('inf')
    patience = 5
    epochs_no_improve = 0
    model_name = args.model.strip(os.sep).split(os.sep)[-1]

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Start time of epoch {epoch}: {datetime.now()}")
        train_loss = train_epoch(model, train_dataset, optimizer, scheduler, epoch, args, scaler)
        train_losses.append(train_loss)

        logger.info(f"Running test set evaluation after {epoch} epochs:")
        eval_loss = evaluate(model, test_dataset, args)
        eval_losses.append(eval_loss)
        if args.decay == "plateau":
            scheduler.step(eval_loss)

        perplexity = math.exp(eval_loss)
        logger.info(f"Perplexity (epoch {epoch}): {perplexity:.3f}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join("checkpoints", f"{model_name}-best")
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_pretrained(checkpoint_path)
            tokenizer.save(os.path.join(checkpoint_path, "tokenizer.json"), pretty=True)
            logger.info(f"Best model saved at: {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Finetuning finished.")
    logger.info(f"Train losses: {train_losses}")
    logger.info(f"Test losses: {eval_losses}")

    # Save loss plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, marker='o', label='Training Loss')
    plt.plot(epochs, eval_losses, marker='s', label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f"{args.model.replace('/', '_')}_loss_plot.png")
    plt.savefig(plot_path)
    logger.info(f"Loss plot saved to {plot_path}")

    # Save losses to CSV
    os.makedirs("metrics", exist_ok=True)
    metrics_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses)+1)),
        "train_loss": train_losses,
        "eval_loss": eval_losses,
    })
    metrics_df.to_csv("metrics/losses.csv", index=False)
    logger.info("Losses saved to metrics/losses.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hugohrban/progen2-small")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--checkpoint_rate", type=int, default=3)
    parser.add_argument("--decay", type=str, choices=["cosine", "linear", "constant", "plateau"], default="cosine")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--save_optimizer", action="store_true", default=False)
    parser.add_argument("--eval_before_train", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(args)