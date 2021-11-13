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

def drawOnImage(image, drawing):

    mask = drawing[:,:,3] # Use alpha channel as mask
    mask_inv = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(image, image, mask=mask_inv)
    foreground = cv2.bitwise_and(drawing, drawing, mask=mask)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGRA2BGR)

    return cv2.add(background, foreground)

def main():

    # Argparse stuff
    parser = argparse.ArgumentParser(description='Augmented reality paint')
    parser.add_argument('-j', '--json', type=str, required=True, help='Full path to json file.')
    parser.add_argument('-usp', '--use_shake_prevention', default=False, action='store_true', help='Use shake prevention functionality. Defaults fo False.')

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
    if cap.isOpened(): 
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        raise Exception('Could not open camera 0.')
    
    # Crate white image
    n_channels = 4
    canvas = np.zeros((height, width, n_channels), np.uint8)
    # canvas.fill(255)

    # Pencil start state
    pencil = {'size': 10, 'color': (0, 0, 255, 255)}
    last_point = None

    # Misc
    norm_threshold = 50

    while True:

        # Read frame
        _, frame = cap.read()

        # Get mask
        mask = cv2.inRange(frame, lower, upper)

        # Get pointer
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        # Get max area mask
        if num_labels > 1:
            
            # Get index ignoring the first one
            index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

            # Get mask and centroid
            mask_max = (labels == index).astype('uint8') * 255
            centroid = centroids[index, :].astype(np.int64)

            # Draw cross
            frame = drawCross(frame, centroid)

            # Compute norm between consecutive points
            norm = (np.linalg.norm(last_point - centroid)) if (args.use_shake_prevention and last_point is not None) else 0

            # Draw line
            if last_point is not None and norm < norm_threshold:
                cv2.line(canvas, last_point, centroid, pencil['color'], pencil['size'])

            # Save last point
            last_point = centroid

            # TODO here?
            cv2.imshow(mask_window, mask_max)

        # Combine frame with drawing
        drawing = drawOnImage(frame, canvas)     

        # Show 
        cv2.imshow(video_window, frame)
        cv2.imshow(canvas_window, drawing)
        
        key = cv2.waitKey(1)

        if key == ord('r'):
            pencil['color'] = (0, 0, 255, 255)
            print('Set pencil color to ' + Fore.RED + 'red' + Style.RESET_ALL)

        elif key == ord('g'):
            pencil['color'] = (0, 255, 0, 255)
            print('Set pencil color to ' + Fore.GREEN + 'green' + Style.RESET_ALL)
        
        elif key == ord('b'):
            pencil['color'] = (255, 0, 0, 255)
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
            canvas.fill(0)
            print('Cleared canvas')
        
        elif key == ord('w'):
            file_name = 'drawing_' + datetime.now().strftime('%a_%b_%m_%H:%M:%S_%Y') + '.png'
            cv2.imwrite(file_name, drawing)
            print('Saved canvas to ' + file_name)
        
        elif key == ord('q'):
            break




           
            



"""
MAIN
"""
if __name__ == '__main__':
    main()