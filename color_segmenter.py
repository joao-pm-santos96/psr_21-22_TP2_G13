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
def onTrackbar(value, limits, channel, trackbar_name, window_name):
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
    is_max = 'max' in trackbar_name
    level = 'max' if is_max else 'min'

    if is_max and (value < limits[channel]['min']):
        cv2.setTrackbarPos(trackbar_name, window_name, limits[channel]['min'])
    elif (not is_max) and (value > limits[channel]['max']):
        cv2.setTrackbarPos(trackbar_name, window_name, limits[channel]['max'])
    else:
        limits[channel][level] = value
        

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
    limits = {'b': {'max': 255, 'min': 0},
    'g': {'max': 255, 'min': 0},
    'r': {'max': 255, 'min': 0}}

    # Create window
    cv2.namedWindow(window_name)

    # Create trackbars
    trackbar_names = {
        'b_min': 'Blue min',
        'b_max': 'Blue max',
        'g_min': 'Green min',
        'g_max': 'Green max',
        'r_min': 'Red min',
        'r_max': 'Red max'
    }

    for channel in 'bgr':
        minc, maxc = f"{channel}_min", f"{channel}_max"
        cv2.createTrackbar(trackbar_names[minc], window_name,   0, 255, partial(onTrackbar, limits=limits, channel=channel, trackbar_name=trackbar_names[minc], window_name=window_name)) 
        cv2.createTrackbar(trackbar_names[maxc], window_name, 255, 255, partial(onTrackbar, limits=limits, channel=channel, trackbar_name=trackbar_names[maxc], window_name=window_name)) 
       

    # Loop
    while True:

        # Get image from camera
        ret, frame = cap.read()

        if not ret:
            print('Skipping...')
            continue

        # Threshold
        lower = np.array([limits[color]['min'] for color in 'bgr'])
        upper = np.array([limits[color]['max'] for color in 'bgr'])

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
