"""
LICENSE PLATE RECOGNITION USING HAAR CASCADE

TO RUN SCRIPT RUNC CMD IN DIRECTORY AND TYPE:
    python palte_detect.py --input YOUR_PATH_TO_IMAGE

"""

import argparse

import cv2
import matplotlib.pyplot as plt

PATH = None
PLATE_CASCADE = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def display(img):
    """
    CONVERT IMAGE COLOR FROM BGR TO RGB AND DISPLAY IT

    Args:
        img (numpy.ndarray): Image object to display
    """
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    plt.show()

def detect_plate(img):
    """
    DETECT PLATE REGION WITH HAAR CASCADE

    Args:
        img (numpy.ndarray): Image object to analyse

    Returns:
        numpy.ndarray: Array of coordinates of detected region(s)
    """
    global PLATE_CASCADE
  
    plate_img = img.copy()
    plate_rects = PLATE_CASCADE.detectMultiScale(plate_img,scaleFactor=1.3, minNeighbors=3) 
        
    return plate_rects

def blur_plate(img, plate_rects):
    """
    BLUR DETECTED REGION OF IMAGE

    Args:
        img (numpy.ndarray): Image object to blur
        plate_rects (numpy.ndarray): Array of coordinates of detected region(s)

    Returns:
        numpy.ndarray: Imgae object with blurred license plates
    """
    plate_img = img.copy()
    roi = img.copy()
    
    for (x,y,w,h) in plate_rects: 
        
        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi,7)
        
        plate_img[y:y+h,x:x+w] = blurred_roi
        
    return plate_img

def main(args):
    """
    MAIN FUNCTION, READ, DETECT AND BLUR THE PLATE

    Args:
        args (argparse.Namespace): Parsed argument with input image path
    """
    global PATH
    
    PATH = args.input
    img = cv2.imread(PATH)
    
    result = blur_plate(img, detect_plate(img))
    
    display(result)
    
def parse_arguments():
    """
    PARSE ARGUMENT WITH PATH OF INPUT IMAGE

    Returns:
        argparse.Namespace: Parsed argument with input image path
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=None)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())
    