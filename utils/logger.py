import os
import json


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
