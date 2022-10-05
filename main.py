from xml.etree.ElementTree import tostring
import cv2

# import matplotlib
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# import matplotlib.image as mpimg
from matplotlib import image as mpimg
import numpy as np
import pandas as pd
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

cap = cv2.VideoCapture(videofile)
# img = cv2.imread("K20-30-0,95_1.jpg")
start_time = datetime.now()
i = 0
z = 0
list1 = [0, 0] * 1000
list2 = [0] * 1000





while True:
    _, img = cap.read()

    # Convert Image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mask Area of Interest
    mask2 = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask2, (555, 390), 230, (255, 255, 255), -1)

    res = cv2.bitwise_and(gray, gray, mask=mask2)

    # Create a threshold
    ret, thresh2 = cv2.threshold(res, 100, 155, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(res, 100, 155, cv2.THRESH_BINARY_INV)

    # titles = ['Original Image','Binary','Binary Inverse']
    # images = [img, thresh, thresh2]

    # for i in range(3):
    # plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    # plt.title(titles[i])
    # plt.xticks([]),plt.yticks([])
    # Plot Points
    # color = (0,0,0)
    # thickness = 40
    # center_coordinates =(550,384)
    # axes_length = (100,50)
    # angle = 0
    # startangle = 0
    # endangle = 360
    # coordinates_links = (450,390)
    # coordinates_rechts = (680,390)
    # coordinates_oben = (555,244)
    # coordinates_unten = (555,530)

    # thresh3 = cv2.drawMarker = (thresh2, center_coordinates, color, cv2.MARKER_CROSS, 50, 200 )
    # thresh3 = cv2.ellipse(thresh2, center_coordinates, axes_length, angle,startangle,endangle,color,thickness)
    # thresh3 = cv2.line(thresh2, coordinates_oben, coordinates_unten, color, 5)
    # thresh3 = cv2.line(thresh2, coordinates_links, coordinates_rechts, color, 5)
    cv2.circle(thresh2, (555, 390), 2, (255, 255, 255), 2)

    contours, heirarchy = cv2.findContours(
        thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_areas = sorted(contours, key=cv2.contourArea)

    mask = np.zeros(gray.shape, np.uint8)

    bounding = cv2.boundingRect(largest_areas[-1])
    (x1, y1, x2, y2) = bounding

    img_contour = cv2.drawContours(
        mask, [largest_areas[-1]], 0, (255, 255, 255, 255), 1
    )

    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    minutes = int(duration / 60)
    seconds = duration % 60

    if keyboard.is_pressed("Spacebar"):
        # workbook = xlsxwriter.Workbook('sample_data4.xlsx')
        # sheet = workbook.add_worksheet()
        # sheet.write('A{}'.format(i + 1), str(x2))
        # sheet.write('B{}'.format(i + 1), str(y2))
        list1[i] = (x2, y2)
        list2[i] = y2
        i = i + 1
        # print(list1[i])
        # workbook = xlsxwriter.Workbook('sample_data4.xlsx')
        # sheet = workbook.add_worksheet()
        # sheet.write('A{}'.format(i + 1), str(x2))
        # sheet.write('B{}'.format(i + 1), str(y2))

    time_elapsed = datetime.now() - start_time
    # print ('%s' % (time_elapsed.seconds))
    s = 0
    z = int(time_elapsed.seconds) - s
    if z > 10:
        s = s + 10
        # print("jetzt")

    # cv2.imshow("kontur", img_contour)

    # Draw boundaries
    for cnt in contours:
        # Draw polgon
        img_polylines = cv2.polylines(img, [largest_areas[-1]], True, (255, 0, 0), 3)

        # rect = cv2.minAreaRect(largest_areas[-1])
        # (x,y), (w,h), angle = rect
        # if cnt[contours] > 2:

        # else:
        # (x1 ,y1 ,x2 ,y2) = (0 ,0 ,0 ,0)

        # ellipse = cv2.fitEllipse(largest_areas[-2])
        # (centerellipse), (axesellipse), (angleellipse) = ellipse
        # print (ellipse)

        # circ = 	cv2.minEnclosingCircle(largest_areas[-2])
        # (center, radius)= circ

        # cv2.circle(img, (int(x),int(y)),5, (0,0,255), -1 )

        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # boxellipse = cv2.boxPoints(ellipse)
        # boxellipse = np.int0(boxellipse)

        # intcenterellipse = np.int0(centerellipse)
        # intaxesellipse = np.int0(axesellipse)
        # intcenter = np.int0(center)
        # intradius = np.int0(radius)

        # cv2.ellipse(img, [box], (70,70,70), 5)
        # cv2.rectangle(img, (int(x-0.5*w),int(y-0.5*h)), (int(x+0.5*w),int(y+0.5*h)), (0,255,0), 1)
        # cv2.polylines (img, [box], True, (255,180,180), 2)
        cv2.rectangle(img, bounding, (0, 180, 180), 2)
        # cv2.circle(img, intcenter, intradius, (0, 0, 255), 2)
        cv2.putText(
            img,
            "Meltpoollaenge: {}".format(round(x2, 1)),
            (100, 50),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )
        cv2.putText(
            img,
            "Meltpoolbreite: {}".format(round(y2, 1)),
            (100, 80),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )
        cv2.putText(
            img,
            "Duration = " + str(minutes) + ":" + str(round(seconds, 2)),
            (100, 740),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )
        cv2.putText(
            img,
            "Time elapsed = {}".format(time_elapsed),
            (100, 710),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )
        cv2.putText(
            img,
            "Time elapsed = {}".format("%s" % (time_elapsed.seconds)),
            (100, 680),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )
        cv2.putText(
            img,
            "Counter = {}".format(i),
            (800, 680),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )

        # y2 Meltpoolbreite
    data = {
        y2,
        x2
    }

    df = pd.DataFrame(data)
    print(df)

    # print("{}".format(y2))

    # cv2.imshow("Gray", gray)
    # cv2.imshow("thresh2", thresh2)
    # cv2.imshow("Result_Mask", res)image.png
    # cv2.imshow("Rect", img_rect)
    # cv2.imshow("Polylines", img_polylines)
    # cv2.imshow("thresh3", thresh3)
    # cv2.imshow("kontur", img_contour)
    # cv2.imshow("thresh", thresh)
    cv2.imshow("Image", img)

    # sheet.write('A{}'.format(i+1), 'Meltpoollaenge: {}'.str(round(x2, 1))
    # sheet.write('B{}'.format(i+1), 'Meltpoolbreite: {}'.str(round(y2, 1))

    # my_wb = openpyxl.Workbook()
    # my_sheet = my_wb.active
    # c1 = my_sheet.cell(row=1, column=1)
    # c1.value = "Laenge"
    # c2 = my_sheet.cell(row=1, column=2)
    # c2.value = "Breite"
    # ci = my_sheet['A{}'.format(i+1)]
    # ci.value = "Meltpoollaenge: {}".format(round(x2, 1))
    # for B2: column = 2 & row = 2.
    # ci = my_sheet['B{}'.format(i+1)]
    # ci.value = "Meltpoolbreite: {}".format(round(y2, 1))
    # my_wb.save("sample_data3.xlsx")

    key = cv2.waitKey(1)

    if key == 27:
        break
# workbook.close()
print(list1)

with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(list1)

cap.release()
cv2.destroyAllWindows()
