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
    white_board = np.ones(frame.shape, np.uint8)  
    white_board.fill(255)

    # Pencil start state
    pencil = {'size': 10, 'color': (0, 0, 255)}

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
            index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

            mask_max = (labels == index).astype('uint8') * 255
            centroid = [int(centroids[index, 0]), int(centroids[index, 1])]

            # Draw cross
            frame = drawCross(frame, centroid)
            
            cv2.imshow(mask_max, mask_max)





        cv2.imshow('Original', frame)
        key = cv2.waitKey(1)

        if key == ord('r'):
            pencil['color'] = (0, 0, 255)
            print('Set pencil color to red')

        elif key == ord('g'):
            pencil['color'] = (0, 255, 0)
            print('Set pencil color to green')
        
        elif key == ord('b'):
            pencil['color'] = (255, 0, 0)
            print('Set pencil color to blue')
        
        elif key == ord('+'):
            # TODO check max size
            pencil['size'] += 1
            print('Increased pencil size to ' + str(pencil['size']))
        
        elif key == ord('-'):
            # TODO check min size
            pencil['size'] -= 1
            print('Decreased pencil size to ' + str(pencil['size']))
        
        elif key == ord('c'):
            print('c')
        
        elif key == ord('-'):
            print('w')
        
        elif key == ord('q'):
            break




           
            



"""
MAIN
"""
if __name__ == '__main__':
    main()