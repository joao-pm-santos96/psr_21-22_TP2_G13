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
"""

"""
FUNCTIONS DEFINITIONS
"""
def onTrackbar(value, limits, channel, is_max, trackbar_name, window_name):
    """Callback for trackbar value changes. This callback limits the maximum (minimum)
    to allways be greater (lower) than the minimum (maximum).

    Args:
        value (int): Value set on the trackbar.
        limits (dictionary): Threshold limits.
        channel (str): Channel.
        isMax (bool): True if we are changing the upper threshold. False otherwise.
        trackbar_name (str): Name of the trackbar.
        window_name (str): Name of the window.
    """    
    levels = ['min', 'max']

    if is_max and (value < limits[channel][levels[not is_max]]):
        cv2.setTrackbarPos(trackbar_name, window_name, limits[channel][levels[not is_max]])
    elif (not is_max) and (value > limits[channel][levels[not is_max]]):
        cv2.setTrackbarPos(trackbar_name, window_name, limits[channel][levels[not is_max]])
    else:
        limits[channel][levels[is_max]] = value
        

def save(limits, file_name = 'limits.json'):
    """Save limits to file

    Args:
        limits (dictionary): Limits.
        file_name (str, optional): File name. Defaults to 'limits.json'.
    """    
    print('Saving to ' + file_name)
    pprint(limits)
    with open(file_name, 'w') as file_handle:
        json.dump(limits, file_handle)

"""
MAIN
"""
if __name__ == '__main__':

    # Init frame and VideoCapture
    window_name = 'Video'
    cap = cv2.VideoCapture(0)
    frame = None

    # Init limits dictionary
    limits = {'limits': {'B': {'max': 255, 'min': 0},
    'G': {'max': 255, 'min': 0},
    'R': {'max': 255, 'min': 0}}}

    # Create window
    cv2.namedWindow(window_name)

    # Create trackbars
    trackbar_names = {'b_min': 'Blue min',
    'b_max': 'Blue max',
    'g_min': 'Green min',
    'g_max': 'Green max',
    'r_min': 'Red min',
    'r_max': 'Red max'}

    cv2.createTrackbar(trackbar_names['b_min'], window_name,   0, 255, partial(onTrackbar, limits=limits['limits'], channel='B', is_max = False, trackbar_name=trackbar_names['b_min'], window_name=window_name)) 
    cv2.createTrackbar(trackbar_names['b_max'], window_name, 255, 255, partial(onTrackbar, limits=limits['limits'], channel='B', is_max = True , trackbar_name=trackbar_names['b_max'], window_name=window_name)) 
    cv2.createTrackbar(trackbar_names['g_min'], window_name,   0, 255, partial(onTrackbar, limits=limits['limits'], channel='G', is_max = False, trackbar_name=trackbar_names['g_min'], window_name=window_name)) 
    cv2.createTrackbar(trackbar_names['g_max'], window_name, 255, 255, partial(onTrackbar, limits=limits['limits'], channel='G', is_max = True , trackbar_name=trackbar_names['g_max'], window_name=window_name)) 
    cv2.createTrackbar(trackbar_names['r_min'], window_name,   0, 255, partial(onTrackbar, limits=limits['limits'], channel='R', is_max = False, trackbar_name=trackbar_names['r_min'], window_name=window_name)) 
    cv2.createTrackbar(trackbar_names['r_max'], window_name, 255, 255, partial(onTrackbar, limits=limits['limits'], channel='R', is_max = True , trackbar_name=trackbar_names['r_max'], window_name=window_name)) 

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
