import sys
import os
import json
from contextlib import contextmanager


@contextmanager
def redirect_stdout(file_path):
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        origin_stdout = sys.stdout
        with open(file_path, "w") as f:
            sys.stdout = f
            yield
            sys.stdout = origin_stdout
    else:
        yield


class StatsLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.stats = {
            "round_stats": [],
            "epoch_stats": [],
        }
        os.makedirs(out_dir, exist_ok=True)

    def log_round(self, duration, loss):
        self.stats["round_stats"].append({"duration": duration, "loss": loss})

    def log_epoch(self, train_loss, val_loss, val_acc):
        self.stats["epoch_stats"].append(
            {"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        )

    def save(self, file_name):
        with open(os.path.join(self.out_dir, file_name), "w") as out_file:
            json.dump(self.stats, out_file, indent=2)
