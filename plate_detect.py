import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = None
PLATE_CASCADE = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)

def detect_plate(img):
    global PLATE_CASCADE
  
    plate_img = img.copy()
    plate_rects = PLATE_CASCADE.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors=3) 
    
    for (x,y,w,h) in plate_rects: 
        cv2.rectangle(plate_img, (x,y), (x+w,y+h), (0,0,255), 4) 
        
    return plate_img

def main(args):
    global PATH
    
    PATH = args.input
    img = cv2.imread(PATH)
    
def parse_arguments():
    '''PARSE INPUT IMAGE'''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())