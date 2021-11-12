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
    lower = np.array([limits['limits']['B']['min'], limits['limits']['G']['min'], limits['limits']['R']['min']])
    upper = np.array([limits['limits']['B']['max'], limits['limits']['G']['max'], limits['limits']['R']['max']])

    # Init frame and VideoCapture
    window_name = 'Video'
    cap = cv2.VideoCapture(0)
    
    # Get initial frame and size
    _, frame = cap.read()
    
    # Crate white image
    white_board = np.ones(frame.shape, np.uint8)  
    white_board.fill(255)

    # Misc
    cross_size = 20

    while True:

        # Read frame
        _, frame = cap.read()

        # Get mask
        mask = cv2.inRange(frame, lower, upper)
        cv2.imshow('Mask', mask)

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
            cv2.line(frame, (centroid[0] - cross_size, centroid[1]), (centroid[0] + cross_size, centroid[1]), (0, 0, 255), 2)
            cv2.line(frame, (centroid[0], centroid[1] - cross_size), (centroid[0], centroid[1] + cross_size), (0, 0, 255), 2)

            cv2.imshow('Mask 2', mask_max)





        cv2.imshow('Original', frame)
        key = cv2.waitKey(1)

        if key == ord('r'):
            print('red')
        elif key == ord('g'):
           print('green')
        elif key == ord('b'):
            print('blue')
        elif key == ord('+'):
                print('+')
        elif key == ord('-'):
            print('-')
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