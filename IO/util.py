import os, torch
from datetime import datetime as dt

def save_model(model_name, model, optimizer, loss, epoch, tokenizer_name, d_model, seq_len, vocab_size, num_layers, base_dir="checkpoints/"):
    check_point = {
        "model_name" : model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "config": {
            "tokenizer_name": tokenizer_name,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_layers": num_layers,
            "seq_len": seq_len,
        }
    }
    os.makedirs(base_dir, exist_ok=True)
    target_dir = base_dir + f"{model_name}/{dt.now().strftime(r"%d-%m-%Y-%H-%M-%S")}/"
    os.makedirs(target_dir, exist_ok=False)
    torch.save(check_point, target_dir + f"{model_name}_checkpoint.pt")
    return