import argparse
from model import KSG_torch

def main(mode="live", model="torch"):
    KSG = None
    if model == "torch":
        KSG = KSG_torch()
        pass
    print(KSG)

    if mode == "live":
        KSG.live_stream()
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Welcome to KSG!')
    parser.add_argument("-o", "--mode", help='operation mode: \t\"live\" or \"file\"')
    parser.add_argument("-m", "--model", help='AI model used: \t\"torch\" or \"onnx\"')
    args = parser.parse_args()

    main(args.mode, args.model)