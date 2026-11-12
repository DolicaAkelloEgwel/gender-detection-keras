# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

import cv2

from detect_gender import detect_gender

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    frame = detect_gender(frame)

    # display output
    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
