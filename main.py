import argparse
import cv2
from model import KSG

def main(single, onnx, path):

    if onnx:
        system = KSG("onnx")
    else: 
        system = KSG("torch")
    if single:
        print("file mode on")
        # system("file", path)
    else: 
        print("live mode on")
        system("live")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Welcome to KSG!')
    parser.add_argument("-o", "--onnx", action='store_true', help='select AI model using \"onnx\"')
    parser.add_argument('-s', '--single', action='store_true', help='only run a inference')
    parser.add_argument('-p', '--path', help='select file inference mode')
    args = parser.parse_args()
    print(args.single, args.onnx, args.path)
    main(args.single, args.onnx, args.path)