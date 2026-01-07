import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import separate


def main():
    parser = argparse.ArgumentParser(description="Reproduce CFRP wavefield separation figures.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        separate.run_training_mode()
    else:
        separate.run_evaluation_mode()


if __name__ == "__main__":
    main()
