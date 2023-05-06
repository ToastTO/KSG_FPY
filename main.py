import argparse
import cv2
from model import KSG

def main(single, onnx, path, display, picam):

    if onnx:
        system = KSG("onnx", display = display, picam = picam)
    else: 
        system = KSG("torch", display = display, picam = picam)
    if single:
        print("file mode not yet implemented")
        # system("file", path)
    else: 
        print("live mode on")
        system("live")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Welcome to KSG!')
    parser.add_argument("-o", "--onnx", action='store_true', help='select AI model using \"onnx\"')
    parser.add_argument('-d', '--display', action='store_true',help='display visualization (bBox)')
    parser.add_argument('-s', '--single', action='store_true', help='only run a inference')
    parser.add_argument('-p', '--path', help='select file inference mode')
    parser.add_argument('--webcam', action='store_false', help='seleting usb web camera as a in put')
    args = parser.parse_args()
    # print(args.single, args.onnx, args.path)
    main(args.single, args.onnx, args.path, args.display, args.webcam)