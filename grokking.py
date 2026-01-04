"""
Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
Re-implementation of the paper for x + y mod 97

Architecture (from Appendix A.1.2):
- Decoder-only transformer with causal attention masking
- 2 layers, width 128, 4 attention heads
- ~4 * 10^5 non-embedding parameters

Optimization:
- AdamW with lr=1e-3, weight_decay=1, beta1=0.9, beta2=0.98
- Linear warmup over first 10 updates
- Batch size 512 (or half training set, whichever smaller)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time


# Hyperparameters
P = 97  # Prime modulus
TRAIN_FRACTION = 0.2
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 128 * 4  # Typical transformer uses 4x
DROPOUT = 0.0  # Paper doesn't mention dropout for main experiments
MAX_STEPS = 20_000
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1.0
BETA1 = 0.9
BETA2 = 0.98
WARMUP_STEPS = 10
LOG_INTERVAL = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class ModularAdditionDataset(Dataset):
    """Dataset for x + y mod p equations."""

    def __init__(self, data):
        """
        data: list of (x, y, result) tuples

        Sequence format: [x, op, y, eq, result]
        We use tokens 0..p-1 for numbers, p for op (+), p+1 for equals (=)
        """
        self.data = data
        self.op_token = P
        self.eq_token = P + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, result = self.data[idx]
        # Sequence: x, +, y, =, result
        input_seq = torch.tensor([x, self.op_token, y, self.eq_token], dtype=torch.long)
        target = torch.tensor(result, dtype=torch.long)
        return input_seq, target


def create_datasets(p=P, train_fraction=TRAIN_FRACTION, seed=42):
    """Create train/val splits for x + y mod p."""
    np.random.seed(seed)

    # Generate all equations
    all_data = []
    for x in range(p):
        for y in range(p):
            result = (x + y) % p
            all_data.append((x, y, result))

    # Shuffle and split
    np.random.shuffle(all_data)
    n_train = int(len(all_data) * train_fraction)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:]

    return ModularAdditionDataset(train_data), ModularAdditionDataset(val_data)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Single transformer decoder block with causal attention."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual (causal)
        seq_len = x.size(1)
        # Create causal mask: positions can only attend to previous positions
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        # Feedforward with residual
        x = x + self.ff(x)
        x = self.ln2(x)
        return x


class DecoderTransformer(nn.Module):
    """Decoder-only transformer for modular arithmetic."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.0):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection (predict the result token)
        self.output = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, seq_len)
        # Embed tokens
        x = self.token_embed(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Get output from last position
        x = x[:, -1, :]  # (batch, d_model)
        logits = self.output(x)  # (batch, vocab_size)

        return logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    """Main training loop with logging."""
    print(f"Device: {DEVICE}")

    # Create datasets
    train_dataset, val_dataset = create_datasets()
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Batch size: min(512, half of training set)
    batch_size = min(BATCH_SIZE, len(train_dataset) // 2)
    print(f"Batch size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    vocab_size = P + 2  # 0..96 for numbers, 97 for +, 98 for =
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(DEVICE)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Optimizer (AdamW with decoupled weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler (linear warmup)
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Logging
    history = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'weight_norm': [],  # For regularization term visualization
    }

    # Training loop
    step = 0
    train_iter = iter(train_loader)

    start_time = time.time()

    pbar = tqdm(total=MAX_STEPS, desc="Training")
    interrupted = False

    try:
        while step < MAX_STEPS:
            model.train()

            # Get batch (cycle through dataset)
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)

            # Logging
            if step % LOG_INTERVAL == 0 or step == 1:
                model.eval()
                with torch.no_grad():
                    # Training metrics (full dataset)
                    train_loss_total = 0
                    train_correct = 0
                    train_total = 0
                    for inputs, targets in train_loader:
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        logits = model(inputs)
                        train_loss_total += F.cross_entropy(logits, targets, reduction='sum').item()
                        preds = logits.argmax(dim=-1)
                        train_correct += (preds == targets).sum().item()
                        train_total += targets.size(0)
                    train_loss = train_loss_total / train_total
                    train_acc = train_correct / train_total * 100

                    # Validation metrics
                    val_loss_total = 0
                    val_correct = 0
                    val_total = 0
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        logits = model(inputs)
                        val_loss_total += F.cross_entropy(logits, targets, reduction='sum').item()
                        preds = logits.argmax(dim=-1)
                        val_correct += (preds == targets).sum().item()
                        val_total += targets.size(0)
                    val_loss = val_loss_total / val_total
                    val_acc = val_correct / val_total * 100

                    # Weight norm (L2) - this is what weight decay regularizes
                    weight_norm = sum(p.pow(2).sum().item() for p in model.parameters())

                    history['step'].append(step)
                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    history['train_acc'].append(train_acc)
                    history['val_acc'].append(val_acc)
                    history['weight_norm'].append(weight_norm)

                    pbar.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'train_acc': f'{train_acc:.1f}%',
                        'val_acc': f'{val_acc:.1f}%'
                    })
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted! Returning results collected so far...")

    pbar.close()
    elapsed = time.time() - start_time
    if interrupted:
        print(f"\nTraining interrupted after {step} steps ({elapsed/60:.1f} minutes)")
    else:
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    return history, model


def plot_results(history):
    """Create plots for training curves."""
    steps = history['step']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training and Validation Loss vs Steps (log scale x-axis)
    ax1 = axes[0, 0]
    ax1.plot(steps, history['train_loss'], label='Train Loss', color='red', alpha=0.8)
    ax1.plot(steps, history['val_loss'], label='Val Loss', color='green', alpha=0.8)
    ax1.axhline(y=np.log(P), color='gray', linestyle='--', label=f'Chance level (ln {P})', alpha=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss vs Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss + Regularization Term vs Steps (dual y-axis)
    ax2 = axes[0, 1]
    train_loss = np.array(history['train_loss'])
    weight_norm = np.array(history['weight_norm'])
    reg_term = WEIGHT_DECAY * weight_norm / 2  # AdamW regularization term

    ax2.plot(steps, train_loss, label='Train Loss', color='red', alpha=0.8)
    ax2.set_xscale('log')
    ax2.set_xlabel('Optimization Steps')
    ax2.set_ylabel('Train Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, reg_term, label=f'Regularization (λ={WEIGHT_DECAY})', color='blue', alpha=0.8)
    ax2_twin.set_ylabel('Regularization Term', color='blue')
    ax2_twin.tick_params(axis='y', labelcolor='blue')

    ax2.set_title('Training Loss + Regularization Term vs Steps')
    ax2.grid(True, alpha=0.3)
    # Combine legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)

    # Plot 3: Training and Validation Accuracy vs Steps
    ax3 = axes[1, 0]
    ax3.plot(steps, history['train_acc'], label='Train Acc', color='red', alpha=0.8)
    ax3.plot(steps, history['val_acc'], label='Val Acc', color='green', alpha=0.8)
    ax3.axhline(y=100/P, color='gray', linestyle='--', label=f'Chance ({100/P:.1f}%)', alpha=0.5)
    ax3.set_xscale('log')
    ax3.set_xlabel('Optimization Steps')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Training and Validation Accuracy vs Steps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])

    # Plot 4: Combined view like in the paper
    ax4 = axes[1, 1]
    ax4.plot(steps, history['train_acc'], label='Train Acc', color='red', alpha=0.8)
    ax4.plot(steps, history['val_acc'], label='Val Acc', color='green', alpha=0.8)
    ax4.set_xscale('log')
    ax4.set_xlabel('Optimization Steps')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Grokking: x + y mod 97 (50% training data)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig('grokking_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved plots to grokking_results.png")


def estimate_training_time():
    """Estimate training time before running."""
    print("=" * 60)
    print("TRAINING TIME ESTIMATE")
    print("=" * 60)

    # Create a small test
    train_dataset, val_dataset = create_datasets()
    batch_size = min(BATCH_SIZE, len(train_dataset) // 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    vocab_size = P + 2
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Time a few batches
    model.train()
    n_warmup = 10
    n_test = 50

    train_iter = iter(train_loader)

    # Warmup
    for _ in range(n_warmup):
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    # Synchronize before timing
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    elif DEVICE.type == 'mps':
        torch.mps.synchronize()

    # Timed run
    start = time.time()
    for _ in range(n_test):
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    elif DEVICE.type == 'mps':
        torch.mps.synchronize()

    elapsed = time.time() - start
    time_per_step = elapsed / n_test

    # Account for logging overhead (roughly every LOG_INTERVAL steps we do full eval)
    # Estimate: logging takes ~10x a regular step
    n_logs = MAX_STEPS // LOG_INTERVAL
    logging_overhead = n_logs * 10 * time_per_step

    total_train_time = MAX_STEPS * time_per_step + logging_overhead

    print(f"Device: {DEVICE}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Batch size: {batch_size}")
    print(f"Training steps: {MAX_STEPS:,}")
    print(f"Time per step: {time_per_step*1000:.2f} ms")
    print(f"Estimated total time: {total_train_time/60:.1f} minutes")
    print("=" * 60)

    return total_train_time


if __name__ == "__main__":
    # First estimate training time
    estimate_training_time()

    print("\nStarting training...")
    history, model = train()

    if not history['step']:
        print("\nNo data collected, exiting without saving.")
        exit(0)

    print("\nGenerating plots...")
    plot_results(history)

    # Save model
    torch.save(model.state_dict(), 'grokking_model.pt')
    print("Saved model to grokking_model.pt")

    # Save history
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    print("Saved training history to training_history.json")
