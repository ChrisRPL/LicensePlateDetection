import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    pass

def parse_arguments():
    '''PARSE INPUT IMAGE'''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())