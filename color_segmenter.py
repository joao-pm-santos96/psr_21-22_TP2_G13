#!/usr/bin/env python3
"""
***DESCRIPTION***
"""

"""
IMPORTS
"""
import cv2
import json
from pprint import pprint
import numpy as np

from functools import partial

"""
TODO
- do not allow max < min
"""

"""
FUNCTIONS DEFINITIONS
"""
def onTrackbar(value, limits, channel, min_max):
    limits[channel][min_max] = value

def save(limits, file_name = 'limits.json'):
    print('Saving to ' + file_name)
    pprint(limits)
    with open(file_name, 'w') as file_handle:
        json.dump(limits, file_handle)

"""
MAIN
"""
if __name__ == '__main__':

    window_name = 'Video'
    cap = cv2.VideoCapture(0)

    limits = {'limits': {'B': {'max': 255, 'min': 0},
    'G': {'max': 255, 'min': 0},
    'R': {'max': 255, 'min': 0}}}

    frame = None

    # Create window
    cv2.namedWindow(window_name)
    cv2.createTrackbar('B min', window_name,   0, 255, partial(onTrackbar, limits=limits['limits'], channel='B', min_max='min')) 
    cv2.createTrackbar('B max', window_name, 255, 255, partial(onTrackbar, limits=limits['limits'], channel='B', min_max='max')) 
    cv2.createTrackbar('G min', window_name,   0, 255, partial(onTrackbar, limits=limits['limits'], channel='G', min_max='min')) 
    cv2.createTrackbar('G max', window_name, 255, 255, partial(onTrackbar, limits=limits['limits'], channel='G', min_max='max')) 
    cv2.createTrackbar('R min', window_name,   0, 255, partial(onTrackbar, limits=limits['limits'], channel='R', min_max='min')) 
    cv2.createTrackbar('R max', window_name, 255, 255, partial(onTrackbar, limits=limits['limits'], channel='R', min_max='max')) 

    # Loop
    while True:

        # Get image from camera
        ret, frame = cap.read()

        if not ret:
            print('Skipping...')
            continue

        # Threshold
        lower = np.array([limits['limits']['B']['min'], limits['limits']['G']['min'], limits['limits']['R']['min']])
        upper = np.array([limits['limits']['B']['max'], limits['limits']['G']['max'], limits['limits']['R']['max']])

        mask = cv2.inRange(frame, lower, upper)
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

        # Display frame
        cv2.imshow(window_name, frame_masked)
        key = cv2.waitKey(5)

        if key == ord('w'):
            save(limits)
            break

        elif key == ord('q'):
            break

    # Finalize
    cap.release()
