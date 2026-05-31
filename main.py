import torch
from datetime import datetime as dt

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Model.Tokenizer import MTokenizer
from Model.MiniLM import MiniLM

from IO.DataLoader import StreamingWindowLoader as Loader
from IO.util import save_model, load_model

from utils.plot_matrix import plot_attention_heatmaps as heatmap


# ============================================================
# Configuration
# ============================================================

DEVICE = "cpu"

WINDOW_SIZE = 128
D_MODEL = 256
VOCAB_SIZE = 50258
NUM_LAYERS = 6
EPOCHS = 50

# CHECKPOINT_PATH = (
#     "checkpoints/MiniLM/01-05-2026-17-29-37/MiniLM_checkpoint.pt"
# )


# ============================================================
# Data
# ============================================================

tokenizer = MTokenizer()

loader = Loader(
    "dataset/tiny_shakespear.txt",
    tokenizer,
    WINDOW_SIZE,
)


# ============================================================
# Model
# ============================================================

lm = MiniLM(
    VOCAB_SIZE,
    D_MODEL,
    NUM_LAYERS,
).to(DEVICE)

loss_fn = CrossEntropyLoss()

optimizer = AdamW(
    lm.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

scheduler = CosineAnnealingLR(
    optimizer,
    EPOCHS,
)

loss_hist = []


# ============================================================
# Utilities
# ============================================================

def sanity_check(
    model: MiniLM,
    inp: str = "To be, or not to be, ",
    limit: int = 20,
    view_heatmap: bool = False,
    do_sample: bool = True,
):
    model.eval()
    model.do_logits_output = False
    model._return_att = True

    text = inp

    for _ in range(limit):
        input_ids = tokenizer.encode(text)["input_ids"]

        probs, weights = model(
            torch.tensor(input_ids).unsqueeze(0)
        )

        if do_sample:
            next_token = torch.multinomial(
                probs.squeeze(0)[-1],
                num_samples=1,
            )
        else:
            next_token = torch.argmax(
                probs.squeeze(0)[-1]
            )

        text += tokenizer.decode(next_token)

    if view_heatmap:
        tokens = [
            tokenizer.decode(token)
            for token in tokenizer.encode(text)["input_ids"][:-1]
        ]

        layer_names = [str(i + 1) for i in range(len(weights))]

        weights = [
            torch.mean(layer[0], dim=0)
            .detach()
            .numpy()
            for layer in weights
        ]

        heatmap(
            weights,
            tokens,
            "Attention weights heatmap",
            layer_names=layer_names,
        )

    model.train()
    model.do_logits_output = True
    model._return_att = False

    print(
        f"[Sanity Check] input={inp!r}\n"
        f"Generated={text[len(inp):]}"
    )


# ============================================================
# Checkpoint
# ============================================================

# lm, optimizer, scheduler, _, _ = load_model(
#     CHECKPOINT_PATH,
#     MiniLM,
#     AdamW,
#     CosineAnnealingLR,
#     "cpu",
#     50,
# )

# sanity_check(lm, "KING:", 32, do_sample=False)


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):
    sanity_check(lm)

    loss_hist.append([])
    lm.train()

    for step, (x, y) in enumerate(loader, start=1):
        x = x.unsqueeze(0).to(DEVICE)
        y = y.unsqueeze(0).to(DEVICE)

        optimizer.zero_grad()

        logits = lm(x)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        loss_hist[epoch].append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            lm.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if step % 100 == 0:
            print(
                f"[{dt.now()}] "
                f"epoch={epoch} step={step}"
            )
            sanity_check(lm)

    sanity_check(lm)

    scheduler.step()

    save_model(
        "MiniLM",
        lm,
        optimizer,
        scheduler,
        loss_hist[epoch][-1],
        epoch,
        "gpt2",
        D_MODEL,
        WINDOW_SIZE,
        VOCAB_SIZE,
        NUM_LAYERS,
    )