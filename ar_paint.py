#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import argparse
import json
import cv2
import numpy as np

from colorama import Fore, Back, Style
from datetime import datetime

"""
TODO
"""

"""
CLASS DEFINITIONS
"""

"""
FUNCTIONS DEFINITIONS
"""
def drawCross(image, center, size = 20, color = (0,0,255), line_width = 2):
    
    cv2.line(image, (center[0] - size, center[1]), (center[0] + size, center[1]), color, line_width)
    cv2.line(image, (center[0], center[1] - size), (center[0], center[1] + size), color, line_width)

    return image

def main():

    # Argparse stuff
    parser = argparse.ArgumentParser(description='Augmented reality paint')
    parser.add_argument('-j', '--json', type=str, required=True, help='Full path to json file.')

    args = parser.parse_args()

    # Read file
    # TODO check if file exists
    with open(args.json) as f:
        limits = json.load(f)

    # Get thresholds
    lower = np.array([limits['limits'][color]['min'] for color in 'bgr'])
    upper = np.array([limits['limits'][color]['max'] for color in 'bgr'])

    # Init frame and VideoCapture
    video_window = 'Video'
    canvas_window = 'Canvas'
    mask_window = 'Mask'

    cap = cv2.VideoCapture(0)
    
    # Get initial frame and size
    _, frame = cap.read()
    
    # Crate white image
    canvas = np.ones(frame.shape, np.uint8)  
    canvas.fill(255)

    # Pencil start state
    pencil = {'size': 10, 'color': (0, 0, 255)}
    last_point = None

    while True:

        # Read frame
        _, frame = cap.read()

        # Get mask
        mask = cv2.inRange(frame, lower, upper)

        # Get pointer
        components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        num_labels = components[0]
        labels = components[1]
        stats = components[2]
        centroids = components[3]

        # Get max area mask
        if num_labels > 1:
            
            # Get index ignoring the first one
            index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

            # Get mask and centroid
            mask_max = (labels == index).astype('uint8') * 255
            centroid = [int(centroids[index, 0]), int(centroids[index, 1])]

            # Draw cross
            frame = drawCross(frame, centroid)

            # Draw line
            if last_point:
                cv2.line(canvas, last_point, centroid, pencil['color'], pencil['size'])

            # Save last point
            last_point = centroid

            cv2.imshow(mask_window, mask_max)

        # Show 
        cv2.imshow(video_window, frame)
        cv2.imshow(canvas_window, canvas)
        
        
        key = cv2.waitKey(1)

        if key == ord('r'):
            pencil['color'] = (0, 0, 255)
            print('Set pencil color to ' + Fore.RED + 'red' + Style.RESET_ALL)

        elif key == ord('g'):
            pencil['color'] = (0, 255, 0)
            print('Set pencil color to ' + Fore.GREEN + 'green' + Style.RESET_ALL)
        
        elif key == ord('b'):
            pencil['color'] = (255, 0, 0)
            print('Set pencil color to ' + Fore.BLUE + 'blue' + Style.RESET_ALL)
        
        elif key == ord('+'):
            pencil['size'] += 1
            print('Increased pencil size to ' + str(pencil['size']))
        
        elif key == ord('-'):
            if pencil['size'] > 2:
                pencil['size'] -= 1
                print('Decreased pencil size to ' + str(pencil['size']))
            else:
                print('Can not decrease any further')

        elif key == ord('c'):
            canvas.fill(255)
            print('Cleared canvas')
        
        elif key == ord('w'):
            file_name = 'drawing_' + datetime.now().strftime('%a_%b_%m_%H:%M:%S_%Y') + '.png'
            cv2.imwrite(file_name, canvas)
            print('Saved canvas to ' + file_name)
        
        elif key == ord('q'):
            break




           
            



"""
MAIN
"""
if __name__ == '__main__':
    main()