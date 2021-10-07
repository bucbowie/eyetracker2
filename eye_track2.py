import sys
import os
try:
    from datetime import datetime as dt
except:
    os.system('pip install datetime')
    from datetime import datetime as dt
from datetime import timedelta
try:
    import cv2
except:
    os.system('pip install python-opencv')
    import cv2
try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np
try:
    import dlib
except:
    os.system('pip install dlib')
    import dlib
try:
    from math import hypot
except:
    os.system('pip install math')
    from math import hypot
try:
    import time
except:
    os.system('pip install time')
    import time
screen = np.zeros((480, 640, 3), dtype="uint8")
kernel = np.ones((9, 9), np.uint8) # Used later on in masking

##############################################################################
#                               FUNCTIONS                                    #
##############################################################################
#--------------------------------------
def shape_to_np(shape, dtype="int"):
#--------------------------------------
# Convert coords to a 2-tuple of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 67):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
#--------------------------------------
def mask_eyes(mask, side):
#--------------------------------------
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

#--------------------------------------
def func_pass(x):
#--------------------------------------
    return None
#--------------------------------------
def midpoint(p1, p2):
#--------------------------------------
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)
#--------------------------------------
def debug_func(*args):
#--------------------------------------
    if DEBUG:
        print(*args)
#--------------------------------------
def draw_circle(event, x, y, flags, param):
#--------------------------------------
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("event: EVENT_LBUTTONDBLCLK")
        global screen
        cv2.circle(screen, (x, y), 10, MAGENTA, -1)
        print("event: EVENT_LBUTTONDBLCLK with coordinates " + str(x) +  "," + str(y))

    if event == cv2.EVENT_MOUSEMOVE:
        image_coordinates = [(x,y)]
        print("event: EVENT_MOUSEMOVE with coordinates " + str(x) +  "," + str(y))

    if event == cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")

    if event == cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")

    if event == cv2.EVENT_RBUTTONDOWN:
        screen = np.zeros((480, 640, 3), dtype="uint8")         #Mouse-20211006


###############################################################################
#                               BEGIN                                         #
###############################################################################
DEBUG = 0                                                   
##############################################################################
#                           BGR COLORS (in order of 1st digit, 2nd...)       #
##############################################################################
BLACK   = (0, 0, 0)
RED     = (0, 0, 255)
ORANGE  = (0, 69, 255)
YELLOW  = (0, 247, 255)
GREEN   = (0, 255, 0)
LIGHT_GREEN = (0, 255, 14)
LIGHT_RED = (2, 53, 255)
CHOCOLATE = (30, 160, 210)
GOLDEN    = (32, 218, 165)
PURPLE = (128, 0, 129)
PINK   = (147, 20, 255)
BLUE   = (255, 0, 0)
MAGENTA= (255, 0, 242)
LIGHT_BLUE = (255, 9, 2)
LIGHT_CYAN = (255, 204,0)
CYAN   = (255, 255, 0)
WHITE  = (255, 255, 255)

##############################################################################
#            FONTS: Type, Scale, Thickness and Marker positions 
##############################################################################
blink_font   = cv2.FONT_HERSHEY_COMPLEX_SMALL   
blink_font_position = (10, 50)                  
blink_font_scale = 1                            
blink_font_thickness =1                         
blink_font_color = (BLUE)                       
center_font = cv2.FONT_HERSHEY_TRIPLEX          
center_font_position = (10, 35)                 
center_font_scale = 1                           
center_font_thickness = 1                       
center_font_color = (GREEN)                     
left_eye_left_side_marker   = (10,200)          
left_eye_right_side_marker  = ( left_eye_left_side_marker[0],left_eye_left_side_marker[1]   + 10) 
right_eye_left_side_marker  = ( left_eye_left_side_marker[0],left_eye_right_side_marker[1]  + 10) 
right_eye_right_side_marker = ( left_eye_left_side_marker[0],right_eye_left_side_marker[1]  + 10) 
gaze_marker                 = ( left_eye_left_side_marker[0],right_eye_right_side_marker[1] + 20) 
direction_marker            = ( left_eye_left_side_marker[0],gaze_marker[1] + 20)
gaze_font = cv2.FONT_HERSHEY_PLAIN              
gaze_font_scale =1                              
gaze_font_thickness = 1                         
gaze_font_color = (PURPLE)                      
threshold_intensity = 70  #(default is 70)      #Parameters-20210923
time_font=cv2.FONT_HERSHEY_COMPLEX_SMALL        #Parameters-20210917
time_font_position=(620,5)                      #Parameters-20210917
time_font_scale = .5                            #Parameters-20210917
time_font_thickness = 1                         #Parameters-20210917
time_font_color = (BLUE)                        #Parameters-20210917
warning_font = cv2.FONT_HERSHEY_PLAIN           #Parameters-20210915
warning_font_position = (120, 30)               
warning_font_scale = 1                          
warning_font_thickness = 1                      
warning_font_color = (RED)                      
#--------------------------------------         
# Sliders defaults                              
#--------------------------------------         
default_slider_centering_gaze = 5               #Parameters-20210929
default_slider_threshold  = 118                 
default_slider_crosshairs = 255                 
default_slider_face_ticks = 255                 
default_x_stringency      = 20                  
default_y_stringency      = 20                  
default_blink             = 6                   #Parameters-20210917
default_slider_single_eye = 40.5                #SingleEye-20210928
default_slider_show_crosshairs = 0 
#--------------------------------------         #Parameters-20210917
# Define the points of the eye                  #Parameters-20210917
#--------------------------------------         #Parameters-20210917
left_eye_facial_points  = [36, 37, 38, 39, 40, 41] #Parameters-20210917
right_eye_facial_points = [42, 43, 44, 45, 46, 47] #Parameters-20210917
chin_point              = [8]                   
center_of_eyes          = [27]                  
#--------------------------------------         #UI-20210916
# Build Settings Window and Sliders             #UI-20210916
#--------------------------------------         #UI-20210916
cv2.namedWindow('Settings', cv2.WINDOW_AUTOSIZE)                                       #UI-20210918
cv2.namedWindow('screen mouse')                                                        #Mouse_20211006
cv2.resizeWindow('Settings',340,370)  #Keep as toy                                     #UI-20210918
cv2.createTrackbar('Eye_thresh','Settings', int(default_slider_single_eye), 255, func_pass) #DEBUG-20210927
cv2.createTrackbar('Dir. Cntl', 'Settings', int(default_slider_centering_gaze), 50, func_pass) #CenterEye-20210929
cv2.createTrackbar('Blink',     'Settings', default_blink,             255, func_pass) #UI-20210917
cv2.createTrackbar('Old-Thresh', 'Settings', default_slider_threshold,  255, func_pass) #UI-20210914
cv2.createTrackbar('X-hairs',   'Settings', default_slider_crosshairs, 255, func_pass) #UI-20210914
cv2.createTrackbar('Face-Marks','Settings', default_slider_face_ticks, 255, func_pass) #UI-20210914
cv2.createTrackbar('X-axis',    'Settings', default_x_stringency,      255, func_pass) #UI-20210916
cv2.createTrackbar('Y-axis',    'Settings', default_y_stringency,      255, func_pass) #UI-20210916
cv2.createTrackbar('Show Xhair', 'Settings', default_slider_show_crosshairs, 1, func_pass)



detector =  dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


#-------------------------------------- #Improve-20200917
# Try and open camera                   
#-------------------------------------- #Improve-20200917
# Set the video input                   #Improve-20200920
#---------------------------------------
videoInput = (1) #"video/eyes_2021-09-20_13.32.03.414371.avi")          
cap = cv2.VideoCapture(videoInput)                                      
if (cap.isOpened() == False):                                           
    debug_func("########################################")              
    debug_func('# Could not open camera. Aborting with no action taken.')
    debug_func("# --> Is the camera plugged in or using built in camera?")
    debug_func("# --> Does the camera work? Have you used it before?")   
    debug_func("# --> Check if camera connections are loose/unattached.")
    debug_func("# --> If plugged in webam, set:                     ")  
    debug_func("#     --> cap = cv2.VideoCapture(x):                ")  
    debug_func("#     --> cv2.VideoCapture(0) = built-in camera     ")  
    debug_func("#     --> cv2.VideoCapture(1) = 1st USB camera      ")  
    debug_func("#     --> cv2.VideoCapture(2) = 2nd USB camera      ")  
    debug_func("########################################")              
    sys.exit(1)                                                         
#--------------------------------------                                 #Improve-20200917
# Get the Image Frame size from the Camera                              
#--------------------------------------                                 #Improve-20200917
img_width = int(cap.get(3))                                             #Improve-20200917
img_height = int(cap.get(4))                                            #Improve-20200917
debug_func("########################################")                  
debug_func("# The camera image is size: ", img_width, " : ", img_height)
debug_func("########################################")                  
debug_func(" ")                                                        
#--------------------------------------                                 #Improve-20200917
# Set up Video VideoCapture                                             
#--------------------------------------                                 #Improve-20200917
current_timestamp = dt.now()                                            
now = current_timestamp.strftime("%Y-%m-%d_%H.%M.%S.%f")                
if str(videoInput).isnumeric():                                         #Improve-20210924
    out = cv2.VideoWriter('video/eyes_' + now + '.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (img_width,img_height)) 
ret, img = cap.read()
#-------------------------------------- 
# Center of image for centering dot     
#-------------------------------------- 
img_center_y=img_height //2 #img.shape[0]//2            
img_center_x=img_width // 2 #img.shape[1]//2            
frame_cnt = 0                                           
start_time =0                                           
end_time = 0                                            
cv2.setMouseCallback('screen mouse', draw_circle)       #Mouse-20211006          

##############################################################################
# MAIN LOGIC LOOP             MAIN LOGIC LOOP                  MAIN LOGIC LOOP
##############################################################################
while(ret):                                                                 
    try:                                                                    
        ret, img = cap.read()
    except:                                                                 
        debug_func("########################################")              
        debug_func('# Could not read any more input. Aborting with no action taken.')
        debug_func("# --> If this is reading a video,it means end of video.") 
        debug_func("# --> If you are using a webcam, ")                     
        debug_func("#     it means issues with the camera.")                
        debug_func("########################################")              
        sys.exit(0)                                                         
    start_time = time.time()                                                
    frame_cnt  += 1                                                         
    thresh = img.copy()
    img2   = img.copy()
                    
    roi = img[ 220:270, 250:305]                            
    roi_rows, roi_cols, _ = roi.shape
    roi2 = roi.copy()
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)        
    Single_eye_threshold = cv2.getTrackbarPos('Eye_thresh', 'Settings') 
    _, threshold = cv2.threshold(gray_roi, Single_eye_threshold, 255, cv2.THRESH_BINARY_INV) 
    blur = cv2.GaussianBlur(threshold,(9,9),0)              
    median = cv2.medianBlur(threshold,7)                    
    cv2.imshow("blur", blur )                               
    cv2.imshow("median", median)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# You can play with these settings
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)                             
    #contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
   
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#   Generating contours - avoid devide by zero bug
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if len(contours) > 0:
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) 
        #contours = max(contours, key=cv2.contourArea) 

        for cnt in contours:                                    
           (x, y, w, h) = cv2.boundingRect(cnt)                                   
           cv2.line(roi, (x + int(w/2), 0),  (x + int(w/2), roi_rows),  ORANGE, 1)
           cv2.line(roi, (0 , y + int(h/2) ), (roi_cols , y + int(h/2)),  ORANGE, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#           Begin LOOP to process the detector's findings
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    for rect in rects:
        shape = predictor(gray, rect)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#UI-20210914
#                     Visual Tracking Customization                         #UI-20210914
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#UI-20210914
        crosshair_shade = cv2.getTrackbarPos('X-hairs', 'Settings')         #UI-20210914
        color_eye_crosshairs   = (0, crosshair_shade, 0)                    
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#Blink-20210913
#                     Experimental Blink Detection                          #Blink-20210913
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#Blink-20210913
        right_eye_left_point = (shape.part(36).x -5, shape.part(36).y  -1)   #Blink-20210913
        right_eye_right_point = (shape.part(39).x -5, shape.part(39).y -1)  #Blink-20210913
        right_eye_center_top = midpoint(shape.part(37), shape.part(38))     #Blink-20210913
        right_eye_center_bottom = midpoint(shape.part(41), shape.part(40))  #Blink-20210913
        

        left_eye_left_point    = (shape.part(42).x -5, shape.part(42).y  -1)   #Blink-20210913
        left_eye_right_point   = (shape.part(45).x -5, shape.part(45).y  -1)   #Blink-20210913
        left_eye_center_top    = midpoint(shape.part(43), shape.part(44) ) #Blink-20210913
        left_eye_center_bottom = midpoint(shape.part(45), shape.part(46) )


        right_eye_center_temp_top = cv2.resize(right_eye_center_top, None, fx=5, fy=5)              #DEBUG-20101016
        cv2.imshow("Reye_Cntr", right_eye_center_temp_top)              #DEBUG-20210106

        
        
        left_eye_center_bottom = midpoint(shape.part(47), shape.part(46))   #Blink-20210913
                
        right_eye_horizontal = hypot((right_eye_left_point[0] - right_eye_right_point[0]), (right_eye_left_point[1] - right_eye_right_point[1])) #Blink-20210913
        left_eye_horizontal  = hypot((left_eye_left_point[0] - left_eye_right_point[0]), (left_eye_left_point[1] - left_eye_right_point[1])) #Blink-20210913
        right_eye_vertical   = hypot((right_eye_center_top[0] - right_eye_center_bottom[0]), (right_eye_center_top[1] - right_eye_center_bottom[1])) #Blink-20210913
        left_eye_vertical    = hypot((left_eye_center_top[0] - left_eye_center_bottom[0]), (left_eye_center_top[1] - left_eye_center_bottom[1])) #Blink-20210913
        
        if right_eye_horizontal + left_eye_horizontal == 0 or ( right_eye_vertical + left_eye_vertical == 0): #Blink-20210928
            ratio = 0                                                                                         #Blink-20210928
        else:                                                                                                 #Blink-20210928
            ratio = (((right_eye_horizontal + left_eye_horizontal) /2 ) / ((right_eye_vertical + left_eye_vertical) / 2) ) #Blink-20210913
        
        #--------------------------------------------                       #Blink-20210917
        # Control Blink sensitivty here                                     #Blink-20210917
        #--------------------------------------------                       #Blink-20210917
        blink_ratio = cv2.getTrackbarPos('Blink', 'Settings')               #Blink-20210917
        if ratio > blink_ratio:                                             #Blink-20210917
            cv2.putText(img, "Blink detected", blink_font_position, blink_font, blink_font_scale, blink_font_color)#Blink-20210913
       
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#Gaze-20210915
#                              Gaze Detection                               #Gaze-20210915
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#Gaze-20210915

        #--------------------------------------------                       #Gaze-20210917
        # Get point of the eyes                                             #Gaze-20210917
        #--------------------------------------------                       #Gaze-20210917
        left_eye_region = np.array([(shape.part(36).x, shape.part(36).y),   #Gaze-20210915
                                    (shape.part(37).x, shape.part(37).y),   #Gaze-20210915
                                    (shape.part(38).x, shape.part(38).y),   #Gaze-20210915
                                    (shape.part(39).x, shape.part(39).y),   #Gaze-20210915
                                    (shape.part(40).x, shape.part(40).y),   #Gaze-20210915
                                    (shape.part(41).x, shape.part(41).y)    #Gaze-20210915
                                    ],np.int32)                             #Gaze-20210915
        right_eye_region = np.array([(shape.part(42).x, shape.part(42).y),  #Gaze-20210916
                                     (shape.part(43).x, shape.part(43).y),  #Gaze-20210916
                                     (shape.part(44).x, shape.part(44).y),  #Gaze-20210916
                                     (shape.part(45).x, shape.part(45).y),  #Gaze-20210916
                                     (shape.part(46).x, shape.part(46).y),  #Gaze-20210916
                                     (shape.part(47).x, shape.part(47).y)   #Gaze-20210916
                                    ],np.int32)                             #Gaze-20210916


        ##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Visualize "Centered" messages on the screen b4 adding screen points 
        ##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
        mid = (shape.part(42).x + shape.part(39).x) // 2                    #Center-20210920
        #---------------------------------------------------
        mid_y_between_eyes = int((shape.part(42).y + shape.part(39).y) / 2) #Center-20210924
        mid_y_between_ears = int((shape.part(0).y  + shape.part(16).y) / 2) #Lateral-20210924
        #---------------------------------------------------
        alignment_to_eyes_x_axis = abs(mid - img_center_x)                  
        alignment_to_eyes_y_axis = abs(mid_y_between_eyes - img_center_y)    
        bottom_of_chin = shape.part(8).x                                         
        x_stringency = cv2.getTrackbarPos('X-axis', 'Settings')             #UI-20210916
        y_stringency = cv2.getTrackbarPos('Y-axis', 'Settings')             #UI-20210916
        if alignment_to_eyes_x_axis < x_stringency:                         
            if alignment_to_eyes_y_axis < y_stringency:                     
                if abs(bottom_of_chin - mid) < 10:                          
                    cv2.putText(img, "Centered and Aligned", center_font_position, center_font, center_font_scale, center_font_color) #Centered-20210913
                    current_timestamp = dt.now()                            
                    now = current_timestamp.strftime("%Y-%m-%d_%H.%M.%S.%f")
                    try:
                        if str(videoInput).isnumeric():                     
                            cv2.putText(img, "Time: " + now, (400,20), time_font, time_font_scale, time_font_color) 
                            out.write(img)                                  
                        else:                                               #Improve-20210924
                            cv2.putText(img, "Time: " + now, (400,30), time_font, time_font_scale, time_font_color) 
                    except:                                                 
                        cv2.putText(img, "Time: " + now, (400,20), time_font, time_font_scale, time_font_color) 
                else:                                                       
                    cv2.putText(img, "Eyes Centered, Face NOT Aligned", warning_font_position, warning_font, warning_font_scale, warning_font_color) 


        height, width, _ = img.shape                                        
        mask_zeros = np.zeros((height,width), np.uint8)                     
      
        #--------------------------------------------                       
        # Draw eye-liner on mask_zeros and img                              
        #--------------------------------------------                       
        cv2.polylines(mask_zeros, [left_eye_region],  True, 255, 1)         
        #cv2.polylines(img, [left_eye_region],  True, 255, 1)                
        cv2.fillPoly(mask_zeros, [left_eye_region], 1)                      
        left_eye_bitwise=cv2.bitwise_and(gray, gray,mask=mask_zeros)        

        cv2.polylines(mask_zeros, [right_eye_region], True, 255, 1)         
        #cv2.polylines(img, [right_eye_region], True, 255, 1)                
        cv2.fillPoly(mask_zeros, [right_eye_region], 1)                     
        right_eye_bitwise=cv2.bitwise_and(gray, gray,mask=mask_zeros)       


        #--------------------------------------------                       #Gaze-20210917
        # Get top, botton and sides of left_eye                             #Gaze-29210917
        #--------------------------------------------                       #Gaze-20210917
        left_min_x = np.min(left_eye_region[:, 0])                          #Gaze-20210915
        left_max_x = np.max(left_eye_region[:, 0])                          #Gaze-20210915
        left_min_y = np.min(left_eye_region[:, 1])                          #Gaze-20210915
        left_max_y = np.max(left_eye_region[:, 1])                          #Gaze-20210915
        #--------------------------------------------                       #Gaze-20210917
        # Get top, botton and sides of right_eye                            #Gaze-29210917
        #--------------------------------------------                       #Gaze-20210917
        right_min_x = np.min(right_eye_region[:, 0])                        #Gaze-20210915
        right_max_x = np.max(right_eye_region[:, 0])                        #Gaze-20210915
        right_min_y = np.min(right_eye_region[:, 1])                        #Gaze-20210915
        right_max_y = np.max(right_eye_region[:, 1])                        #Gaze-20210915

        ##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Visualize left/right eye movement messages on the screen          
        ##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        left_eye_gray = left_eye_bitwise[left_min_y:left_max_y, left_min_x: left_max_x ]              
        _, threshold_left_eye = cv2.threshold(left_eye_gray, threshold_intensity, 255, cv2.THRESH_BINARY)              

        right_eye_gray = right_eye_bitwise[right_min_y:right_max_y, right_min_x: right_max_x ]        
        _, threshold_right_eye = cv2.threshold(right_eye_gray, threshold_intensity, 255, cv2.THRESH_BINARY)            
        
        #===================================================================#DEBUG-20210923
        threshold_left_eye = cv2.dilate(threshold_left_eye, kernel, 5)      #DEBUG-20210923
        #===================================================================#DEBUG-20210923

        left_eye_height, left_eye_width = threshold_left_eye.shape                                          
        left_eye_left_side_threshold = threshold_left_eye[0:left_eye_height, 0: int(left_eye_width/2)]      #LeftRight-20210921
        left_eye_right_side_threshold = threshold_left_eye[0: left_eye_height, int(left_eye_width/2): left_eye_width] #LeftRight-20210921
        left_eye_left_side_white = cv2.countNonZero(left_eye_left_side_threshold)                           #LeftRight-20210921
        left_eye_right_side_white = cv2.countNonZero(left_eye_right_side_threshold)                         #LeftRight-20210921
       
        right_eye_height, right_eye_width = threshold_right_eye.shape                                       
        right_eye_left_side_threshold  = threshold_right_eye[0: right_eye_height, 0: int(right_eye_width/2)]#LeftRight-20210921
        right_eye_right_side_threshold = threshold_right_eye[0: right_eye_height, int(right_eye_width/2): right_eye_width] #LeftRight-20210921
        right_eye_left_side_white = cv2.countNonZero(right_eye_left_side_threshold)                         #LeftRight-20210921
        right_eye_right_side_white = cv2.countNonZero(right_eye_right_side_threshold)                       #v-20210921

        cv2.putText(img, "Leye_left_sclera:   "  + str(left_eye_left_side_white),  left_eye_left_side_marker,   warning_font, warning_font_scale/2, warning_font_color) 
        cv2.putText(img, "Leye_right_sclera:  "  + str(left_eye_right_side_white), left_eye_right_side_marker,  warning_font, warning_font_scale/2, warning_font_color) 
        cv2.putText(img, "Reye_left_sclera:  "   + str(right_eye_left_side_white), right_eye_left_side_marker,  warning_font, warning_font_scale/2, warning_font_color) 
        cv2.putText(img, "Reye_right_sclera: "   + str(right_eye_right_side_white),right_eye_right_side_marker, warning_font, warning_font_scale/2, warning_font_color) 
        
        left_eye_gaze_ratio = left_eye_left_side_white - left_eye_right_side_white     #LeftRight-20210921
        right_eye_gaze_ratio = right_eye_left_side_white -  right_eye_right_side_white #LeftRight-20210921

        Center_gaze_threshold = cv2.getTrackbarPos('Dir. Cntl', 'Settings')            #CenterEye-20210929
        if (right_eye_gaze_ratio - left_eye_gaze_ratio) == 0:                          #CenterEye-20210929
            gaze_ratio = 0                                                             #CenterEye-20210929
        else:                                                                          #CenterEye-20210929
             gaze_ratio = (left_eye_gaze_ratio + right_eye_gaze_ratio) / 2             #CenterEye-20210929

        if gaze_ratio >= -2 and (gaze_ratio <=2):                                      #CenterEye-20210929
            gaze_font_color = (0, 255, 0)                                              #CenterEye-20210929     
            direction = 'CENTER'                                                       #CenterEye-20210929
        elif gaze_ratio < Center_gaze_threshold:                                       #CenterEye-20210929
            gaze_font_color = (0 ,0, 255)                                              #CenterEye-20210929
            direction = 'RIGHT'                                                        #CenterEye-20210929
        else: #gaze_ratio >  Center_gaze_threshold:                                    #CenterEye-20210929
            gaze_font_color = (255,0 , 0)                                              #CenterEye-20210929
            direction = 'LEFT'                                                         #CenterEye-20210929
                                                                                       #CenterEye-20210929

        
        cv2.putText(img, "Dir. Indicator: " + str(gaze_ratio),gaze_marker, gaze_font, gaze_font_scale, gaze_font_color) #Gaze-20210922
        cv2.putText(img, "Gaze Direction: " + direction,direction_marker,  gaze_font, gaze_font_scale, gaze_font_color) #Gaze-20210922
        gaze_font_color = (PURPLE)                                                  #LeftRight-20210924


        threshold_left_eye = cv2.resize(threshold_left_eye, None, fx=5, fy=5)
        just_left_eye  = cv2.resize(left_eye_gray, None, fx=5, fy=5)
        just_right_eye = cv2.resize(right_eye_gray, None, fx=5, fy=5) #DEBUG-20210921
     
        left_eye  = img[left_min_y: left_max_y, left_min_x: left_max_x]     #Gaze-20210920
        right_eye = img[right_min_y: right_max_y, right_min_x: right_max_x] #Gaze-20210920
        left_eye  = cv2.resize(left_eye,  None, fx=5, fy=5)                 #Gaze-20210920
        right_eye = cv2.resize(right_eye, None, fx=5, fy=5)                 #Gaze-20210920
        left_eye_gray  = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)         #Gaze-20210920
        right_eye_gray = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)        #Gaze-20210920

        height_left_eye, width_left_eye, _  = left_eye.shape                #Gaze-20210916
        height_right_eye, width_right_eye, _ = right_eye.shape              #Gaze-20210916
                  
        shape = shape_to_np(shape)
        
        #cv2.rectangle(img, (225,225), (305,255),(255, 0, 0), 2)             #Improve-20210928
        cv2.rectangle(img, (225,220), (305,255),(255, 0, 0), 2)             #Improve-20210928
        face_ticks = cv2.getTrackbarPos('Face-Marks', 'Settings')           #UI-20210914
        for (x, y) in shape[0:36]: #[0:48]                                  #BUGFIX-20210924
            cv2.circle(img, (x, y), 1, (face_ticks, 0, 0), -1)
            cv2.line(img, shape[8], shape[27], (124, 90, 85), 1)            
            cv2.line(img, shape[0], shape[16], (CYAN), 1)            #Lateral-20210930
            cv2.line(img, shape[27], ((int(img_width/2)),img_height), (0, 255, 0), thickness=1, lineType=8, shift=0) 
            cv2.line(img, shape[0],   (int(img_width/2), int((img_height/2))), (124, 0, 255), thickness=1, lineType=8, shift=0) #Lateral-20210924
            #if DEBUG:

        ##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        # Visualize the centering dots in middle of screen                  
        ##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            cv2.circle(img, (mid, mid_y_between_eyes), 12, (255, 0, 255), -1)   
            cv2.circle(img,(img_center_x,img_center_y), 12, (118,118,118), -1)  
     
        ##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #              visualize the cross-hairs on the eyes                
        ##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        Crosshair_state    = cv2.getTrackbarPos('Show Xhair', 'Settings')
        if int(Crosshair_state) > 0:
            left_eye_hor_line  = cv2.line(img, left_eye_left_point,  left_eye_right_point,   (color_eye_crosshairs), 1) 
            left_eye_ver_line  = cv2.line(img, left_eye_center_top,  left_eye_center_bottom, (color_eye_crosshairs), 1) 
            right_eye_hor_line = cv2.line(img, right_eye_left_point, right_eye_right_point,  (color_eye_crosshairs), 1)
            right_eye_ver_line = cv2.line(img, right_eye_center_top, right_eye_center_bottom,(color_eye_crosshairs), 1) 
        #DEBUG##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #DEBUG# Visualize the eye points being tracked                            
        #DEBUG##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        if DEBUG:
            cv2.polylines(img, [left_eye_region], True, (0,0,255),1)
            cv2.polylines(img, [right_eye_region], True, (0,0,255),1)
        if DEBUG:                                                          
            cv2.putText(img, "DEBUG: " , (1,10), time_font, time_font_scale, BLUE) 
            cv2.putText(img, " ON", (45,10), time_font,  time_font_scale, RED,thickness=2)
            cv2.putText(img, "Align Chin to green vertical line", (shape[8][0] , (shape[8][1] - 5)), time_font, time_font_scale, BLACK) 
            cv2.putText(img, "Align Gray and Pink Dots", ((img_center_x), (img_center_y + 10)), time_font, time_font_scale, PINK) 
            cv2.putText(img, "You can adjust X-axis and Y-axis", ((img_center_x), (img_center_y + 20)), time_font, time_font_scale, BLACK) 
            cv2.putText(img, "setting to adjust sensitivity.", ((img_center_x), (img_center_y + 30)), time_font, time_font_scale, BLACK) 
            cv2.putText(img, "Align face to pink line", (shape[0][0] , (shape[0][1] - 5)), time_font, time_font_scale, (CYAN)) 
        else:
            cv2.putText(img, "DEBUG: " , (1,10), time_font, time_font_scale, BLUE)
            cv2.putText(img, " OFF " , (45,10), time_font, time_font_scale, GREEN) 

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#SuperImpose-20210929
    # Size roi2 and superimpose onto "img"                             #SuperImpose-20210929
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#SuperImpose-20210929
    roi2 = cv2.resize(roi2, None, fx=3, fy=3)                          #SuperImpose-20210929
    img[90:190,10:175,:] = roi2[0:100,0:165,:]                         #SuperImpose-20210929
    cv2.rectangle(img, (10,90), (175,190), BLUE,2)                     #SuperImpose-20210929
    end_time = time.time()                                             
    img_duration = end_time - start_time                               
    cv2.putText(img, "Frame Processing Speed: " + str(img_duration), (400,10), time_font, time_font_scale, time_font_color) 
    cv2.imshow('eyes', img)
    cv2.imshow('screen mouse', screen)
    
    
#######################################
# ENDING LOGIC
#######################################
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
out.release()                                                       
cv2.destroyAllWindows()
