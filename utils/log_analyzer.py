import argparse
import re
import os
from collections import defaultdict
import json


def analyze_log(log_path, ana_path):
    kernel_time_pattern = (
        r"Triton autotuning for function\s+"
        r"`(?P<func>[\w_]+)`\s+"
        r"with\s+`(?P<params>\([^`]+\))`\s+"
        r"on config\s+`[^`]+`\s+"
        r"took\s+(?P<time>[\d.]+)ms"
    )
    # best_config_pattern_w_time = (
    #     r"Triton autotuning for function `(?P<func>\w+)` "
    #     r"with `(?P<params>.+?)` "
    #     r"finished after (?P<time>[0-9.]+)s"
    # )
    best_config_pattern = (
        r"Triton autotuning for function `(?P<func>\w+)` "
        r"with `(?P<params>.+?)` finished"
    )
    epoch_pattern = r"^\s*epoch\s*:\s*(\d+)\s*$"
    round_pattern = r"^\s*round\s*:\s*(\d+)\s*$"
    regex = {
        "kernel_time": re.compile(kernel_time_pattern, re.DOTALL),
        "best_config": re.compile(best_config_pattern),
        "epoch": re.compile(epoch_pattern),
        "round": re.compile(round_pattern),
    }
    ana_result = []

    with open(log_path, "r") as f:
        for line in f:
            for pattern in regex:
                res = regex[pattern].search(line)
                if not res:
                    continue
                if pattern == "epoch":
                    ana_result.append([])
                elif pattern == "round":
                    ana_result[-1].append(
                        defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
                    )
                    ana_result[-1][-1]["total_kernel_time_ms"] = 0
                elif pattern == "kernel_time":
                    func_name = res.group("func")
                    params_str = res.group("params")
                    time_ms = float(res.group("time"))
                    ana_result[-1][-1]["total_kernel_time_ms"] += time_ms
                    ana_result[-1][-1][func_name][pattern][params_str] = time_ms
                elif pattern == "best_config":
                    func_name = res.group("func")
                    params_str = res.group("params")
                    # time_s = float(res.group("time"))
                    ana_result[-1][-1][func_name][pattern] = params_str
    os.makedirs(os.path.dirname(ana_path), exist_ok=True)
    with open(ana_path, "w") as f:
        json.dump(ana_result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Analyze log data")
    parser.add_argument("--log-path", type=str, help="Log file path")
    parser.add_argument(
        "--ana-path", type=str, help="File path to save the analysis result"
    )
    args = parser.parse_args()
    analyze_log(args.log_path, args.ana_path)


if __name__ == "__main__":
    main()
