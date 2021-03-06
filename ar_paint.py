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
import random
import sys
import math
from time import time

from colorama import Fore, Back, Style
from datetime import datetime



"""
TODO
"""

"""
CLASS DEFINITIONS
"""
class MouseHandler:
    def __init__(self, pencil, flushShape):
        self.pencil = pencil
        self.drawing = False
        self.last_point = None
        self.flushShape = flushShape

    def onMouseClick(self, event,x,y,flags,param):

        # activate mouve drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x,y)
            if self.pencil['shape']=='.':
                self.pencil['last_point'] = (x,y)
        
        # modify last_point
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.last_point = (x,y)  
            
        # unset drawing and last_point
        elif event == cv2.EVENT_LBUTTONUP:
            
            self.flushShape()
            self.drawing = False
            self.last_point = None

def distanceOf2Points(p1,p2):
        """Compute the distance between two 2D points.

        Args:
            p1 (tuple): Point 1.
            p2 (tuple): Point 2.

        Returns:
            float: Distance.
        """    
        
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5 if (p1 is not None and p2 is not None) else None

def drawShape(image, pencil, centroid):
        """Draw a shape ('circle' or 'rectangle') into an image.

        Args:
            image (np.ndarray): Image to be drawn.
            pencil (dict): Caracteristics of the shapes to be drawn.
            centroid (tuple): Position of the shape.
        """    
        shape = pencil["shape"]

        if shape == 's':
            cv2.rectangle(image, pencil["last_point"], centroid, pencil['color'], pencil['size'])
        elif shape == 'e': 
            distance_x = abs(centroid[0]-pencil["last_point"][0])
            distance_y = abs(centroid[1]-pencil["last_point"][1])
            cv2.ellipse(image, pencil["last_point"], (distance_x, distance_y), 0, 0, 360, color=pencil['color'], thickness=pencil['size']) 

        return image 

def drawOnImage(image, drawing):
        """Helper function to draw on top of another image.

        Args:
            image (np.ndarray): Background image.
            drawing (np.ndarray): Foreground image.

        Returns:
            np.ndarray: Merged images.
        """    

        mask = drawing[:,:,3] # use alpha channel as mask
        mask_inv = cv2.bitwise_not(mask)

        background = cv2.bitwise_and(image, image, mask=mask_inv)
        foreground = cv2.bitwise_and(drawing, drawing, mask=mask)
        
        return cv2.add(background, foreground)


class ImageHandler:

    def __init__(self, shake_prevention = False, mirror = False, camera_mode = False, paint_mode = False): # to avoid using global variables

        self.shake_prevention = shake_prevention
        self.mirror = mirror
        self.camera_mode = camera_mode
        self.paint_mode = paint_mode

        self.pencil_lower_limits = None
        self.pencil_upper_limits = None 

        self.video_window = 'Video'
        self.canvas_window = 'Canvas'
        self.mask_window = 'Mask'
        self.goal_paint_window = 'Goal Paint'

        self.capture = None
        self.img_width = None
        self.img_height = None
        self.n_channels = 4

        self.canvas = None
        self.persistent_background = None
        self.shape_canvas = None
        self.goal_paint = None
        self.centroid = None

        self.mouse_handler = None

        self.pencil = {'size': 10, 'color': (255, 0, 0, 255), "last_point": None, "shape": '.'}
        self.drawing = None

        self.shake_threshold = 150        

        self.key_counter = {None: None}

    def flushCurrentShape(self):
        self.pencil['shape'] = '.'
        self.canvas = drawOnImage(self.canvas, self.shape_canvas)
        self.pencil['last_point'] = None 
        self.shape_canvas.fill(0)   

    def getLimitsFromFile(self, file):
        """Get the threshold limits from file

        Args:
            file (string): File path.
        """        

        try:
            f = open(file)
            limits = json.load(f)

            # get thresholds
            self.pencil_lower_limits = np.array([limits['limits'][color]['min'] for color in 'bgr'])
            self.pencil_upper_limits = np.array([limits['limits'][color]['max'] for color in 'bgr'])

        except:
            print(f"Error reading the file '{file}': {sys.exc_info()[1]}")
            exit(1)

    def startWindows(self):
        """Start all cv2 widnows.
        """        

        cv2.namedWindow(self.video_window)
        cv2.namedWindow(self.mask_window)
        cv2.namedWindow(self.canvas_window)

    def setCanvasCallback(self):
        """Set canvas callback method.
        """        
        self.mouse_handler = MouseHandler(self.pencil, self.flushCurrentShape)
        cv2.setMouseCallback(self.canvas_window, self.mouse_handler.onMouseClick)

    def startVideoCapture(self, index=0):
        """Start video capture.

        Args:
            index (int, optional): Camera index. Defaults to 0.

        Returns:
            bool: Success.
        """         
        self.capture = cv2.VideoCapture(index)

        # get initial frame and size
        if self.capture.isOpened(): 
            self.img_width  = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        return self.capture.isOpened()

    def startCanvas(self):
        """Start user canvas.
        """        

        # create white canvas
        self.canvas = np.zeros((self.img_height, self.img_width, self.n_channels), np.uint8)
        self.persistent_background = np.zeros((self.img_height, self.img_width, self.n_channels), np.uint8)
        self.shape_canvas = np.zeros((self.img_height, self.img_width, self.n_channels), np.uint8)
        self.drawing = np.ones((self.img_height, self.img_width, self.n_channels), np.uint8) * 255

        if not self.camera_mode:
            self.canvas.fill(255)

        try:
            if self.paint_mode:

                # read the image to be painted resize it to camera
                self.goal_paint = cv2.imread(self.paint_mode, cv2.IMREAD_COLOR)
                self.goal_paint = cv2.resize(self.goal_paint, (self.img_width, self.img_height))

                # compute mask for almost white areas
                lower = np.array([220 for color in 'bgr']) 
                upper = np.array([256 for color in 'bgr']) 
                mask = cv2.inRange(self.goal_paint, lower, upper)

                # get stats
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
                
                self.goal_paint.fill(0)

                # paint each region randomly (fair distribution)
                fair_random_modifier = math.ceil(num_labels/3)
                region_color = [(255,0,0),(0,255,0),(0,0,255)]*fair_random_modifier
                random.shuffle(region_color)
                for idx in range(1,num_labels): 
                    self.goal_paint[labels==idx] = region_color[idx-1]

                # force pixels to black
                self.persistent_background[labels==0] = (0,0,0,255)    
        except:
            
            print(f"Error reading the file '{self.paint_mode}': {sys.exc_info()[1]}")
            exit(1)

    def handleKey(self, key):       
        """Handle key press.

        Args:
            key (int): ASCII code of key pressed.

        Returns:
            bool: True if quit, False otherwise.
        """        

        # check if drawing rectangle or circle
        shape = chr(key) if key in [ord('s'), ord('e')] else None  

        if shape:
            if self.pencil['shape'] != '.':
                self.flushCurrentShape()
            else:
                self.pencil['shape'] = shape

        elif key == ord('r'):
            if self.pencil['color'] != (0, 0, 255, 255):
                self.pencil['color'] = (0, 0, 255, 255)
                print(f"Set pencil color to {Fore.RED}red{Style.RESET_ALL}")

        elif key == ord('g'):
            if self.pencil['color'] != (0, 255, 0, 255):
                self.pencil['color'] = (0, 255, 0, 255)
                print(f"Set pencil color to {Fore.GREEN}green{Style.RESET_ALL}")
        
        elif key == ord('b'):
            if self.pencil['color'] != (255, 0, 0, 255):
                self.pencil['color'] = (255, 0, 0, 255)
                print(f"Set pencil color to {Fore.BLUE}blue{Style.RESET_ALL}")

        elif key == ord('+'):
            self.pencil['size'] += 1
            print(f"Increased pencil size to {self.pencil['size']}")
        
        elif key == ord('-'):
            if self.pencil['size'] > 2:
                self.pencil['size'] -= 1
                print(f"Decreased pencil size to {self.pencil['size']}")
            else:
                print("Can not decrease any further")

        elif key == ord('c'):
            if self.camera_mode: # make all pixels transparent
                self.canvas.fill(0) 
            else: # paint all pixels as white
                self.canvas.fill(255)
            print('Cleared canvas')
        
        elif key == ord('w'):
            file_name = 'drawing_' + datetime.now().strftime('%a_%b_%m_%H:%M:%S_%Y') + '.png'
            cv2.imwrite(file_name, self.drawing) 
            print('Saved canvas to ' + file_name)

        elif key == ord('q'):
            return True  

        return False
    
    def drawCrosshair(self, image, center, size = 20, color = (0,0,255), line_width = 2):
        """Function to draw a crosshair into an image

        Args:
            image (np.ndarray): Image where the crosshair will be drawn.
            center (tuple): Center position of the crosshair in the image.
            size (int, optional): Size of the crosshair, in pix. Defaults to 20.
            color (tuple, optional): Color of the crosshair. Defaults to (0,0,255).
            line_width (int, optional): With of the crosshair. Defaults to 2.

        Returns:
            np.ndarray: Image with the crosshair drawn.
        """    
        
        cv2.line(image, (center[0] - size, center[1]), (center[0] + size, center[1]), color, line_width)
        cv2.line(image, (center[0], center[1] - size), (center[0], center[1] + size), color, line_width) 

        return image



    def getCrosshair(self, frame):
        """Draw the crosshait on the image.

        Args:
            frame (np.ndarray): Image to be drawn.
            display (bool, optional): Do an imshow of the image. Defaults to True.

        Returns:
            tuple: Centroid of the crosshair.
            mask: Mask of region.
        """        
        centroid = None
        mask_max = np.zeros(frame.shape, np.uint8)
        
        # get mask of drawing tool
        mask = cv2.inRange(frame, self.pencil_lower_limits, self.pencil_upper_limits)

        # get pointer
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        if num_labels > 1:
                
            # get index ignoring the first one
            index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            # get mask and centroid
            mask_max = (labels == index).astype('uint8') * 255
            #print(centroids[index, :].astype(np.int32))
            centroid = tuple(centroids[index, :].astype(np.int32))
        
        return (self.mouse_handler.last_point if self.mouse_handler.drawing else centroid, mask_max)

    def getPaintingAccuracy(self):
        """Compute the score of the paiting.

        Returns:
            float: Accuracy/score.
        """        

        # attribute the painting am accuracy score
        black_template = np.array([0,0,0]) 
        # black pixels are not valid for the accuracy 
        valid_indexes = np.bitwise_not(np.all(self.goal_paint==black_template,axis=2))
        # all_hits contains all correcly painted pixeis
        all_hits = np.all(self.goal_paint==self.canvas[:,:,:3],axis=2)
        # all_valid_hits contains all correcly painted pixeis that are not black
        all_valid_hits = np.logical_and(all_hits, valid_indexes)
        # accuracy is calculated by all valid hits divided by the total pixels that can be painted
        return all_valid_hits.sum()/valid_indexes.sum() if valid_indexes.sum()>0 else 1     

    def drawLine(self, canvas):
        
        # compute distance between points
        last_point = self.pencil["last_point"]
    
        # calculate distances
        norm = distanceOf2Points(last_point, self.centroid)

        # drawing condition
        draw_cond = last_point is not None and (not self.shake_prevention or norm < self.shake_threshold)

        # draw line
        if self.pencil["shape"] == '.' and draw_cond:
            cv2.line(canvas, last_point, self.centroid, self.pencil['color'], self.pencil['size'])

        return canvas

    def run(self, fps=25):

        # introduce fps feature
        # only update window states FPS times per second
        # this prevents the cv2 windows from getting block on continuous key-press events
        time_counter = time()

        # loop
        while self.capture.isOpened():
            
            current_time = time() 
            # update every 1/FPS seconds 
            if (current_time-time_counter)>1/fps:
                # time_count always increases its value by multiples of 1/fps 
                while current_time-time_counter>1/fps:
                    time_counter = 2*current_time - time_counter - 1/fps

                # read frame
                _, frame = self.capture.read()
                

                # mirror image for better compreension
                if self.mirror:
                    frame = cv2.flip(frame, 1) # 1 is code for horizontal flip  

                # compute centroid of pencil and draw it
                self.centroid, pencil_mask = self.getCrosshair(frame)
                
                # convert frame to BGRA
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

                if self.centroid is not None:        

                    frame = self.drawCrosshair(frame, self.centroid, color=(0,0,255)) if not self.mouse_handler.drawing else frame

                    if self.pencil['shape'] == '.':
                        # draw line
                        self.canvas = self.drawLine(self.canvas)

                        # save last point
                        self.pencil["last_point"] = self.centroid

                    else:
                        # clear shape canvas
                        self.shape_canvas.fill(0)

                        if self.pencil["last_point"] is None:
                            self.pencil["last_point"] = self.centroid

                        # redraw shape with the new position, color and thickness on shape_canvas
                        self.shape_canvas = drawShape(self.shape_canvas, self.pencil, self.centroid)                                  

                # combine everything
                self.drawing = drawOnImage(frame, self.canvas) if self.camera_mode else self.canvas.copy()
                
                # crosshair need to be added manually in case of camera_mode is off
                if self.centroid is not None and not self.camera_mode and not self.mouse_handler.drawing:
                    self.drawing = self.drawCrosshair(self.drawing, self.centroid, color=(0,0,255))
                
                if self.pencil['shape'] != '.' and self.shape_canvas.any():
                    # add temporary shape drawing
                    self.drawing = drawOnImage(self.drawing, self.shape_canvas)

                elif self.pencil['shape'] == '.' and self.shape_canvas.any(): 
                    self.pencil["last_point"] = None               

                if self.paint_mode:                    
                    # Always draw the black lines on top
                    self.drawing = drawOnImage(self.drawing, self.persistent_background)

                    # Compute accuracy
                    accuracy = self.getPaintingAccuracy()   
                    
                    # copy goal paint to allow dynamic accuracy display
                    goal_display = self.goal_paint.copy()

                    cv2.putText(goal_display,f"{accuracy*100:.2f}%",(15, goal_display.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,255),3,cv2.LINE_AA)
                    cv2.imshow(self.goal_paint_window, goal_display)            

                if frame is not None:
                    # show capture 
                    cv2.imshow(self.video_window, frame)

                if self.drawing is not None:
                    # show drawing
                    cv2.imshow(self.canvas_window, self.drawing)
                    
                if pencil_mask is not None:
                    # show mask
                    cv2.imshow(self.mask_window, pencil_mask)
            
            if self.handleKey(cv2.waitKey(1)):
                break 

        self.capture.release()

"""
FUNCTIONS DEFINITIONS
"""

def welcomeMessage():
    """Print welcome message.
    """    

    # multi-line legible text format
    welcome_text =f"""Hello! Welcome to our AR Painting app!
    In this amazing app, you can draw either with your mouse or, even better, with any object segmented with color_segmenter.py!

    Here are the all important controls:

    {Style.BRIGHT}[COLORS]{Style.RESET_ALL}
    {Fore.RED}    'r' - red {Style.RESET_ALL}
    {Fore.GREEN}    'g' - green{Style.RESET_ALL}
    {Fore.BLUE}    'b' - blue{Style.RESET_ALL}

    {Style.BRIGHT}[LINE]{Style.RESET_ALL}
        '+' - increase size
        '-' - decrease size

    {Style.BRIGHT}[SHAPES]{Style.RESET_ALL}
        's' - square/rectangle
        'e' - ellipse/circle

        To use this mode, you press once to start the shape, and press again to end it.

    {Style.BRIGHT}[MISC]{Style.RESET_ALL}
        'c' - clear the canvas
        'w' - write the canvas disk
        'q' - quit 

    Let your inner Picasso take the best of you! :)

    (c) PSR 21-22 G13
    """.replace("\n    ","\n")

    print(welcome_text)

def main():

    # input arguments handling using argparse library
    parser = argparse.ArgumentParser(description='Augmented reality paint')
    parser.add_argument('-j', '--json', type=str, required=True, help='Full path to json file.')
    parser.add_argument('-usp', '--use_shake_prevention', default=False, action='store_true', help='Use shake prevention functionality. Defaults fo False.')
    parser.add_argument('-cm', '--cameramode', default=False, action='store_true', help='Use a camera video instead feed a whiteboard to draw.')
    parser.add_argument('-paint', '--paintmode', type=str, help='Use paintmode on a given image')
    parser.add_argument('-m', '--mirror', default=False, action='store_true', help='Mirror camera input horizontally')

    # args is a dictionary
    args = parser.parse_args()

    # print welcome message
    welcomeMessage()

    image_handler = ImageHandler(
        shake_prevention=args.use_shake_prevention, 
        mirror=args.mirror,
        camera_mode=args.cameramode,
        paint_mode=args.paintmode
    )

    image_handler.getLimitsFromFile(args.json)     

    image_handler.startWindows()

    image_handler.startVideoCapture()
    
    image_handler.startCanvas()

    image_handler.setCanvasCallback()

    image_handler.run()        

if __name__ == '__main__':
    main()
