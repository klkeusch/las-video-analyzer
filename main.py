__author__ = "Yannick Ruppert, Sven-David Otto, Klaus Keusch"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Yannick Ruppert, Sven-David Otto, Klaus Keusch"]
__license__ = "tba"
__version__ = "0.1"
__maintainer__ = "Klaus Keusch"
__email__ = "klaus.keusch@htwsaar.de"
__status__ = "Development"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import openpyxl
import time
# from timer import Timer
from datetime import datetime
import keyboard
import xlsxwriter
import csv

import os, sys

args = sys.argv[1:]
print(os.path.dirname(os.path.abspath(sys.argv[0])))

for arguments in args:
    print(arguments)

videofile = "{}".format(args[0])

#img = cv2.imread("K20-30-0,95_1.jpg")
#cap = cv2.VideoCapture('22-30-0,95.mp4')
# cap = cv2.VideoCapture('testvideo.mp4')
# cap = cv2.VideoCapture('C:/Users/sotto/PycharmProjects/IP_Laser_pycv/testvideo.mp4')
cap = cv2.VideoCapture(videofile)

start_time = datetime.now()
i = 0
z = 0
list1 = [0, 0] * 1000
list2 = [0] * 1000

s = 0

while True:
    ret, img = cap.read()

    if not cap.isOpened():
        print("No Input")
        break

    if not ret:
        print("End of File?")
        break

    # Convert Image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Mask Area of Interest
    mask2 = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask2, (555, 390), 230, (255, 255, 255), -1)

    res = cv2.bitwise_and(gray, gray, mask=mask2)

    # Create a threshold
    ret, thresh2 = cv2.threshold(res, 100, 155, cv2.THRESH_BINARY)



    cv2.circle(thresh2, (555, 390), 2, (255, 255, 255), 2)


    contours, heirarchy = cv2.findContours(thresh2 ,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    largest_areas = sorted(contours, key=cv2.contourArea)

    mask = np.zeros(gray.shape, np.uint8)

    bounding = cv2.boundingRect(largest_areas[-1])
    (x1, y1, x2, y2) = bounding

    img_contour = cv2.drawContours(mask, [largest_areas[-1]], 0, (255, 255, 255, 255), 1)

    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    minutes = int(duration / 60)
    seconds = duration % 60



    #if keyboard.is_pressed('Spacebar'):

        # list1[i] = (x2, y2)
        # list2[i] = y2
        # i = i + 1


    # INSERT YOUR CODE

    time_elapsed = datetime.now() - start_time
    #print ('%s' % (time_elapsed.seconds))

    z = int(time_elapsed.seconds) - s
    if z > 1:
        s = s + 1
        print('jetzt')

        list1[i] = (x2, y2)
        list2[i] = y2
        i = i + 1



    #Draw boundaries
    for cnt in contours:
        #Draw polgon
        img_polylines = cv2.polylines(img, [largest_areas[-1]], True, (255, 0, 0), 3)


        cv2.rectangle(img, bounding, (0, 180, 180), 2)
        #cv2.circle(img, intcenter, intradius, (0, 0, 255), 2)
        cv2.putText(img,"Meltpoollaenge: {}".format(round(x2,1)), (100, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255))
        cv2.putText(img, "Meltpoolbreite: {}".format(round(y2,1)), (100, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        cv2.putText(img, 'Duration = ' + str(minutes) + ':' + str(round(seconds, 2)), (100, 740), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        cv2.putText(img, 'Time elapsed = {}'.format(time_elapsed), (100, 710), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        cv2.putText(img, 'Time elapsed = {}'.format('%s' % (time_elapsed.seconds)), (100, 680), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))
        cv2.putText(img, 'Counter = {}'.format(i), (800, 680), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255))




    cv2.imshow("Image", img)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print(list1)

with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(list1)

cap.release()
cv2.destroyAllWindows()
