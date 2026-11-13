# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

import sys

import cv2

from gender_detection import GenderDetection

gd = GenderDetection()

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    sys.exit()

# loop through frames
while webcam.isOpened():
    # read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        sys.exit()

    frame = gd.detect_gender(frame)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
