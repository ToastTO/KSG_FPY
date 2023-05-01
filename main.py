import argparse
from model.py import KSG_torch

def main(mode="live", model="torch"):
    KSG = None
    if model == "torch":
        KSG = KSG_torch()
    print(KSG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Welcome to KSG!')
    parser.add_argument("--mode", help='operation mode: \t\"live\" or \"file\"')
    parser.add_argument("--model", help='AI model used: \t\"torch\" or \"onnx\"')
    args = parser.parse_args()

    main(args.mode, args.model)