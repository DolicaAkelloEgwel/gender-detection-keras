# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

import argparse
import sys

import cv2

from gender_detection import GenderDetection

args = argparse.ArgumentParser()

args.add_argument(
    "-m",
    "--model",
    required=False,
    help="path to optional custom model file",
)

group = args.add_mutually_exclusive_group(required=True)
group.add_argument(
    "-i",
    "--image",
    help="path to the image file to analyse",
)
group.add_argument(
    "-w",
    "--webcam",
    action="store_true",
    help="analyse webcam input",
)

args = args.parse_args()

if args.model:
    gd = GenderDetection(args.model)
else:
    gd = GenderDetection()


def _detect_gender_and_show(image):
    image = gd.detect_gender(image)
    cv2.imshow("gender detection", image)


# read input image
if args.image:
    image = cv2.imread(args.image)

    if image is None:
        print("Could not read input image")
        sys.exit()

    _detect_gender_and_show(image)

    # press any key to close window
    cv2.waitKey()

    # save output
    cv2.imwrite("gender-detection.jpg", image)

if args.webcam:
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

        _detect_gender_and_show(frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # release resources
    webcam.release()

# release resources
cv2.destroyAllWindows()
