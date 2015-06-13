#!/usr/bin/python
import sys
import time
import math
import datetime
import serial
import cv2

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size

min_size = (30,30)
image_scale = 0.5
haar_scale = 1.3
min_neighbors = 4
haar_flags = 0#cv2.CASCADE_SCALE_IMAGE)

# For OpenCV image display
WINDOW_NAME = 'FaceTracker'

def track(img, threshold=100):
    '''Accepts BGR image and optional object threshold between 0 and 255 (default = 100).
       Returns: (x,y) coordinates of centroid if found
                (-1,-1) if no centroid was found
                None if user hit ESC
    '''
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # convert color input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # scale input image for faster processing
    small_img = cv2.resize(gray, (0,0), fx=image_scale, fy=image_scale) 
    small_img = cv2.equalizeHist(small_img)

    center = (-1,-1,-1)
    if(cascade):
        t = cv2.getTickCount()
        # HaarDetectObjects takes 0.02s
        faces = cascade.detectMultiScale(small_img, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size, flags=haar_flags)

        t = cv2.getTickCount() - t
        if len(faces) > 0:
            x1, y1, w, h = faces[0]
            x2 = x1 + w
            y2 = y1 + h
            # the input to Haar was resized, so scale the
            # bounding box of each face and convert it to two Points
            pt1 = (int(x1 / image_scale), int(y1 / image_scale))
            pt2 = (int(x2 / image_scale), int(y2 / image_scale))
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), 2)

            # get the xy corner co-ords, calc the center location
            centerx = x1+((x2-x1)/2)
            centery = y1+((y2-y1)/2)
            h = y2 - y1
            center = (centerx, centery, h*3)

    if cmp(center, (-1,-1,-1)) == 0:
        center = None

    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(1) & 0xFF == 27:
        center = None
    return center


if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    running = True

    while running:
        running, img = cam.read()
        if track(img) is None:
            break

    cam.release()
    cv2.destroyAllWindows()
