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

from colorama import Fore, Back, Style
from datetime import datetime

"""
TODO
"""

"""
CLASS DEFINITIONS
"""
class MouseHandler:
    def __init__(self, canvas, pencil):
        self.canvas = canvas
        self.pencil = pencil
        self.drawing = False
        self.last_point = None

    def onMouseClick(self, event,x,y,flags,param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x,y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.canvas, self.last_point, (x,y), self.pencil['color'], self.pencil['size'])
                self.last_point = (x,y)                

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

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
        self.goal_paint = None
        self.centroid = None

        self.mouse_handler = None

        self.pencil = {'size': 10, 'color': (0, 0, 255, 255), "last_point": None, "shape": '.'}
        self.drawing = None

        self.shake_threshold = 150        

        self.key_counter = {None: None}

    def getLimitsFromFile(self, file):
        """Get the threshold limits from file

        Args:
            file (string): File path.
        """        

        try:
            f = open(file)
            limits = json.load(f)

            # Get thresholds
            self.pencil_lower_limits = np.array([limits['limits'][color]['min'] for color in 'bgr'])
            self.pencil_upper_limits = np.array([limits['limits'][color]['max'] for color in 'bgr'])

        except:
            print(f"Erro a ler o ficheiro {file}!")
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
        self.mouse_handler = MouseHandler(self.canvas, self.pencil)
        cv2.setMouseCallback(self.canvas_window, self.mouse_handler.onMouseClick)

    def startVideoCapture(self, index=0):
        """Start video capture.

        Args:
            index (int, optional): Camera index. Defaults to 0.

        Returns:
            bool: Success.
        """         
        self.capture = cv2.VideoCapture(index)

        # Get initial frame and size
        if self.capture.isOpened(): 
            self.img_width  = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        return self.capture.isOpened()


    def startCanvas(self):
        """Start user canvas.
        """        

        # Create white canvas
        self.canvas = np.zeros((self.img_height, self.img_width, self.n_channels), np.uint8)
        self.persistent_background = np.zeros((self.img_height, self.img_width, self.n_channels), np.uint8)

        if not self.camera_mode:
            self.canvas.fill(255)

        try:
            if self.paint_mode:

                # Read the image to be painted and build a mask
                self.goal_paint = cv2.imread(self.paint_mode, cv2.IMREAD_COLOR)
                self.goal_paint = cv2.resize(self.goal_paint, (self.img_width, self.img_height))

                # Build mask
                lower = np.array([200 for color in 'bgr']) 
                upper = np.array([256 for color in 'bgr']) 
                mask = cv2.inRange(self.goal_paint, lower, upper)

                # Get stats
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
                
                # Paint each region randomly
                self.goal_paint.fill(0)
                region_color = [255,0,0]
                for idx in range(1,num_labels):
                    random.shuffle(region_color)
                    self.goal_paint[labels==idx] = region_color

                self.persistent_background[labels==0,3] = 255    
        except:
            print(f"Erro a ler o ficheiro {self.paint_mode}!")
            exit(1)

    def handleKey(self, key):       
        """Handle key press.

        Args:
            key (int): ASCII code of key pressed.

        Returns:
            bool: True if quit, False otherwise.
        """        

        # Check if drawing rectangle or circle
        shape = chr(key) if key in [ord('s'), ord('e')] else None        
      
        if shape and self.pencil["last_point"]:
            if shape == self.pencil["shape"]:
                self.drawShape(self.canvas, self.pencil, self.centroid)
                self.pencil["shape"] = '.'
                self.pencil["last_point"] = None
            else:
                self.pencil["shape"] = shape
                self.pencil["last_point"] = self.centroid



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

    def drawOnImage(self, image, drawing):
        """Helper function to draw on top of another image.

        Args:
            image (np.ndarray): Background image.
            drawing (np.ndarray): Foreground image.

        Returns:
            np.ndarray: Drawn image.
        """    

        mask = drawing[:,:,3] # Use alpha channel as mask
        mask_inv = cv2.bitwise_not(mask)

        background = cv2.bitwise_and(image, image, mask=mask_inv)
        foreground = cv2.bitwise_and(drawing, drawing, mask=mask)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_BGRA2BGR)

        return cv2.add(background, foreground)

    def distanceOf2Points(self, p1,p2):
        """Compute the distance between two 2D points.

        Args:
            p1 (tuple): Point 1.
            p2 (tuple): Point 2.

        Returns:
            float: Distance.
        """    
        
        return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

    def drawShape(self, image, pencil, centroid):
        """Draw a shape ('circle' or 'rectangle') into an image.

        Args:
            image (np.ndarray): Image to be drawn.
            pencil (dict): Caracteristics of the shepes to be drawn.
            centroid (tuple): Position of the shape.
        """    
        shape = pencil["shape"]

        if shape == 's':
            cv2.rectangle(image, pencil["last_point"], centroid, pencil['color'], pencil['size'])
        elif shape == 'e': 
            distance_x = abs(centroid[0]-pencil["last_point"][0])
            distance_y = abs(centroid[1]-pencil["last_point"][1])
            cv2.ellipse(image, pencil["last_point"], (distance_x, distance_y), 0, 0, 360, color=pencil['color'], thickness=pencil['size'])  

    def getCrosshair(self, frame, display=True):
        """Draw the crosshait on the image.

        Args:
            frame (np.ndarray): Image to be drawn.
            display (bool, optional): Do an imshow of the image. Defaults to True.

        Returns:
            tuple: Centroid of the crosshair.
        """        

        # Get mask of drawing tool
        mask = cv2.inRange(frame, self.pencil_lower_limits, self.pencil_upper_limits)

        # Get pointer
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        if num_labels > 1:
                
            # Get index ignoring the first one
            index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

            # Get mask and centroid
            mask_max = (labels == index).astype('uint8') * 255

            if display:
                cv2.imshow(self.mask_window, mask_max)

            return tuple(centroids[index, :].astype(np.int32))

        return None

    def paintingMode(self):
        """Compute the score of the paiting.

        Returns:
            float: Accuracy/score.
        """        

        # Give the painting a score
        black_template = np.array([0,0,0]) 
        valid_indexes = np.bitwise_not(np.all(self.goal_paint==black_template,axis=2))

        all_hits = np.all(self.goal_paint==self.canvas[:,:,:3],axis=2)
    
        all_valid_hits = np.logical_and(all_hits, valid_indexes)

        # Accuracy
        return all_valid_hits.sum()/valid_indexes.sum()            

    def run(self):

        # Loop
        while self.capture.isOpened():

            # Read frame
            _, frame = self.capture.read()

            # Mirror image for better compreension
            if self.mirror:
                frame = cv2.flip(frame, 1) # code for horizontal  

            self.centroid = self.getCrosshair(frame, display=True)       

            if self.centroid is not None:

                # If not using mouse, draw the crosshair
                if not self.mouse_handler.drawing:
                    frame = self.drawCrosshair(frame, self.centroid, color=self.pencil['color'])
                
                last_point = self.pencil["last_point"] # help legibility

                # Compute norm between consecutive points
                norm = self.distanceOf2Points(last_point, self.centroid) if (self.shake_prevention and last_point is not None) else 0

                # Draw line
                if self.pencil["shape"] == '.':
                    if last_point is not None and norm < self.shake_threshold and not self.mouse_handler.drawing:
                        cv2.line(self.canvas, last_point, self.centroid, self.pencil['color'], self.pencil['size'])

                    # Save last point
                    self.pencil["last_point"] = self.centroid

            # Combine frame with drawing
            self.drawing = self.drawOnImage(frame, self.canvas) if self.camera_mode else self.canvas.copy() 

            if self.pencil["shape"] != '.' and self.centroid:
                self.drawShape(self.drawing, self.pencil, self.centroid)

            self.drawing = self.drawOnImage(self.drawing[:,:,:3], self.persistent_background)

            if self.paint_mode:
                accuracy = self.paintingMode()   
                
                goal_display = self.goal_paint.copy()

                cv2.putText(goal_display,f"{accuracy*100:.2f}%",(10, goal_display.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2,cv2.LINE_AA)
                cv2.imshow(self.goal_paint_window, goal_display)            

            # Show 
            cv2.imshow(self.video_window, frame)
            cv2.imshow(self.canvas_window, self.drawing)

            if self.handleKey(cv2.waitKey(1)):
                break 

        self.capture.release()

"""
FUNCTIONS DEFINITIONS
"""

def welcomeMessage():
    """Print welcome message.
    """    

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

    # Argparse stuff
    parser = argparse.ArgumentParser(description='Augmented reality paint')
    parser.add_argument('-j', '--json', type=str, required=True, help='Full path to json file.')
    parser.add_argument('-usp', '--use_shake_prevention', default=False, action='store_true', help='Use shake prevention functionality. Defaults fo False.')
    parser.add_argument('-cm', '--cameramode', default=False, action='store_true', help='Use a camera video instead feed a whiteboard to draw.')
    parser.add_argument('-paint', '--paintmode', type=str, help='Use paintmode on a given image')
    parser.add_argument('-m', '--mirror', default=False, action='store_true', help='Mirror camera input horizontally')

    args = parser.parse_args()

    # Print welcome message
    welcomeMessage()

    image_handler = ImageHandler(shake_prevention=args.use_shake_prevention, 
    mirror=args.mirror,
    camera_mode=args.cameramode,
    paint_mode=args.paintmode)

    image_handler.getLimitsFromFile(args.json)     

    image_handler.startWindows()

    image_handler.startVideoCapture()
    
    image_handler.startCanvas()

    image_handler.setCanvasCallback()

    image_handler.run()        

if __name__ == '__main__':
    main()