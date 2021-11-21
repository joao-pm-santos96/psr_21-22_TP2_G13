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

def distanceOf2Points(p1,p2):
    
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5


def drawShape(image, pencil, centroid):
    shape = pencil["shape"]

    if shape == 's':
        cv2.rectangle(image, pencil["last_point"], centroid, pencil['color'], pencil['size'])
    elif shape == 'e': 
        cv2.ellipse(image, pencil["last_point"], (abs(centroid[0]-pencil["last_point"][0]),abs(centroid[1]-pencil["last_point"][1]) ), 0, 0, 360, color=pencil['color'], thickness=pencil['size'])


class ImageHandler:
    def __init__(self): # to avoid using global variables
        pass


class MouseHandler:
    def __init__(self, canvas, pencil):
        self.canvas = canvas
        self.pencil = pencil
        self.drawing = False
        self.last_point = None

    def onMouseClick(self, event,x,y,flags,param):
        # print(event)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x,y)


        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.canvas, self.last_point, (x,y), self.pencil['color'], self.pencil['size'])
                self.last_point = (x,y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            


def main():

    # Argparse stuff
    parser = argparse.ArgumentParser(description='Augmented reality paint')
    parser.add_argument('-j', '--json', type=str, required=True, help='Full path to json file.')
    parser.add_argument('-usp', '--use_shake_prevention', default=False, action='store_true', help='Use shake prevention functionality. Defaults fo False.')
    parser.add_argument('-cm', '--cameramode', default=False, action='store_true', help='Use a camera video instead feed a whiteboard to draw.')
    parser.add_argument('-paint', '--paintmode', type=str, help='Use paintmode on a given image')
    parser.add_argument('-m', '--mirror', default=False, action='store_true', help='Mirror camera input horizontally')

    args = parser.parse_args()

    # Read file
    # TODO check if file exists
    
    try:
        f = open(args.json)
        limits = json.load(f)
    except:
        print(f"Erro a ler o ficheiro {args.json}!")
        exit(1)

    # Get thresholds
    lower = np.array([limits['limits'][color]['min'] for color in 'bgr'])
    upper = np.array([limits['limits'][color]['max'] for color in 'bgr'])

    # Init frame and VideoCapture
    video_window = 'Video'
    canvas_window = 'Canvas'
    mask_window = 'Mask'
    goal_paint_window = 'goal_paint'

    cv2.namedWindow(video_window)
    cv2.namedWindow(mask_window)
    cv2.namedWindow(canvas_window)

    cap = cv2.VideoCapture(0)
    
    # Get initial frame and size
    if cap.isOpened(): 
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        raise Exception('Could not open camera 0.')
    
    # Create white image
    n_channels = 4
    canvas = np.zeros((height, width, n_channels), np.uint8)
    persistent_background = np.zeros((height, width, n_channels), np.uint8)
    
    if not args.cameramode:
        canvas.fill(255)


    try:
        if args.paintmode:
            goal_paint = cv2.imread(args.paintmode)
            lower = np.array([250 for color in 'bgr'])
            upper = np.array([256 for color in 'bgr'])
            mask = cv2.inRange(goal_paint, lower, upper)

            # Get pointer
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
            
            goal_paint.fill(0)
            for idx in range(1,num_labels):
                labels[labels==idx] = (idx%3)+2
            
            goal_paint[labels==2,0] = 255
            goal_paint[labels==3,1] = 255
            goal_paint[labels==4,2] = 255

            persistent_background[labels==0,3] = 255         
   
    except:
        print(f"Erro a ler o ficheiro {args.paintmode}!")
        exit(1)
   

    # Pencil start state
    pencil = {'size': 10, 'color': (0, 0, 255, 255), "last_point": None, "shape": '.'}    

    mouse_handler = MouseHandler(canvas, pencil)

    cv2.setMouseCallback(canvas_window, mouse_handler.onMouseClick)
    # Misc
    norm_threshold = 150

    while True:

        # Read frame
        _, frame = cap.read()

        if args.mirror:
            frame = cv2.flip(frame, 1) # code for horizontal 
          

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
            centroid = tuple(centroids[index, :].astype(np.int32))
            # Draw cross

            if not mouse_handler.drawing:
                frame = drawCross(frame, centroid, color=pencil['color'] )
         
            last_point = pencil["last_point"] # help legibility
            
            # Compute norm between consecutive points
            norm = distanceOf2Points(last_point, centroid) if (args.use_shake_prevention and last_point is not None) else 0

            # Draw line
            if pencil["shape"] == '.':
                if last_point is not None and norm < norm_threshold and not mouse_handler.drawing:
                    cv2.line(canvas, last_point, centroid, pencil['color'], pencil['size'])

                # Save last point
                pencil["last_point"] = centroid
            else:
                drawShape(frame, pencil, centroid)


            cv2.imshow(mask_window, mask_max)

        # Combine frame with drawing
        drawing = drawOnImage(frame, canvas) if args.cameramode else canvas.copy()
        
        drawing = drawOnImage(drawing[:,:,:3], persistent_background)
        if args.paintmode:
            black_template = np.array([0,0,0]) 
            valid_indexes = np.bitwise_not(np.all(goal_paint==black_template,axis=2))
       
            all_hits = np.all(goal_paint==canvas[:,:,:3],axis=2)
            
            all_valid_hits = np.logical_and(all_hits, valid_indexes)
          
            goal_display = goal_paint.copy()
            
            accuracy = all_valid_hits.sum()/valid_indexes.sum()
            cv2.putText(goal_display,f"{accuracy*100:.2f}%",(10, goal_display.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
            cv2.imshow(goal_paint_window, goal_display)
        # Show 
        cv2.imshow(video_window, frame)
        cv2.imshow(canvas_window, drawing)
        
        
        # Key controls
        key = cv2.waitKey(1)

        # Check if drawing rectangle or circle
        shape = chr(key) if key in [ord('s'), ord('e')] else None

        if shape and pencil["last_point"]:
            if shape == pencil["shape"]:
                drawShape(canvas, pencil, centroid)
                pencil["shape"] = '.'
                pencil["last_point"] = None
            else:
                pencil["shape"] = shape
                pencil["last_point"] = centroid

        elif key == ord('r'):
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
            if args.cameramode: # make all pixels transparent
                canvas[:,:,:] = 0 
            else: # paint all pixels as white
                canvas.fill(255)
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