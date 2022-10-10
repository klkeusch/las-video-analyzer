__author__ = "Yannick Ruppert, Sven-David Otto, Klaus Keusch"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Yannick Ruppert, Sven-David Otto, Klaus Keusch"]
__license__ = "tba"
__version__ = "0.1"
__maintainer__ = "Klaus Keusch"
__email__ = "klaus.keusch@htwsaar.de"
__status__ = "Development"

import sys
from itertools import zip_longest
import cv2
import numpy as np
from datetime import datetime
import csv

# Command line arguments
if len(sys.argv) != 2:
    print("Check your input file! \ne.g. <program.py> <videofile.type>")
    exit()

args = sys.argv[1:]

# Vars
start_time = datetime.now()
videofile = "{}".format(args[0])
sampling_rate = 1
measurment_number = 0
frame_rate = 0
frame_count_sampling = 0
length_arr = [0] * 1000
height_arr = [0] * 1000

# GUI
position_text_left_side = 25
position_text_top = 25
position_text_bottom = 680
position_text_right_side = 750

# Main
print("Loaded video file: {} at {}".format(videofile, start_time))

# Source
cap = cv2.VideoCapture(videofile)

if not cap.isOpened():
    print("Cannot open media!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
minutes = int(duration / 60)
seconds = duration % 60

print("Frame rate: {} FPS" .format(int(fps)))

while True:
    ret, img = cap.read()

    if not ret:
        print("End of File?")
        break

    # Convert Image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mask Area of Interest
    mask2 = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask2, (555, 390), 230, (255, 255, 255), -1)

    res = cv2.bitwise_and(gray, gray, mask=mask2)

    # Create a threshold
    ret, thresh2 = cv2.threshold(res, 100, 155, cv2.THRESH_BINARY)

    cv2.circle(thresh2, (555, 390), 2, (255, 255, 255), 2)

    contours, hierarchy = cv2.findContours(
        thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_areas = sorted(contours, key=cv2.contourArea)

    mask = np.zeros(gray.shape, np.uint8)

    bounding = cv2.boundingRect(largest_areas[-1])
    (x1, y1, x2, y2) = bounding

    img_contour = cv2.drawContours(
        mask, [largest_areas[-1]], 0, (255, 255, 255, 255), 1
    )

    time_elapsed = datetime.now() - start_time

    frame_rate = int(fps)
    if frame_count_sampling >= (frame_rate*sampling_rate):
        length_arr[measurment_number] = x2
        measurment_number += 1
        height_arr[measurment_number] = y2
        print("Measuring...({})".format(measurment_number))
        frame_count_sampling = 0
    else: frame_count_sampling +=1

    # Draw boundaries
    for cnt in contours:
        # Draw polgon
        img_polylines = cv2.polylines(img, [largest_areas[-1]], True, (255, 0, 0), 3)

        cv2.rectangle(img, bounding, (0, 180, 180), 2)

        cv2.putText(
            img,
            "Stop measurement: press q",
            (position_text_left_side, position_text_top),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )

        cv2.putText(
            img,
            "Meltpool length: {}".format(round(x2, 1)),
            (position_text_left_side, position_text_top + 35),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )

        cv2.putText(
            img,
            "Meltpool height: {}".format(round(y2, 1)),
            (position_text_left_side, position_text_top + 60),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )

        cv2.putText(
            img,
            "Duration: " + str(minutes) + ":" + str(round(seconds, 2)),
            (position_text_right_side, position_text_bottom + 30),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )

        cv2.putText(
            img,
            "Time elapsed: {}".format(time_elapsed),
            (position_text_left_side, position_text_bottom + 30),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )

        cv2.putText(
            img,
            "Time elapsed (sec): {}".format("%s" % (time_elapsed.seconds)),
            (position_text_left_side, position_text_bottom),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )

        cv2.putText(
            img,
            "Measurements = {}".format(measurment_number),
            (position_text_right_side, position_text_bottom),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 0, 255),
        )

    cv2.imshow("Active video file: {}".format(videofile), img)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        print("Aborted by user!")
        break

measured_data = [length_arr, height_arr]
export_measured_data = zip_longest(*measured_data, fillvalue="")

csv_filename = "{}.csv".format(videofile)
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(("Meltpool length", "Meltpool height"))
    writer.writerows(export_measured_data)

print("Media playback time in seconds: {}".format("%s" % (time_elapsed.seconds)))
print("Number of measurements taken: {}".format(measurment_number))
print("Measurements saved in file: {}".format(csv_filename))

cap.release()
cv2.destroyAllWindows()
