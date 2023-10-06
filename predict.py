import argparse

from engine.predictor import Predictor
from train import get_args_parser


def main():
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model_path = 'run/box_model/detr-r50-e632da11.pth'
    predictor = Predictor(model_path, args)
    image_path = 'datasets/images/11.png'
    output = predictor.predict(image_path)
    print(output)

if __name__ == '__main__':
    main()