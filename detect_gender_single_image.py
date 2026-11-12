# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

import cv2

# import necessary packages
from detect_gender import detect_gender

# read input image
image = cv2.imread("sample_input.jpg")

if image is None:
    print("Could not read input image")
    exit()

image = detect_gender(image)

# display output
cv2.imshow("gender detection", image)

# press any key to close window
cv2.waitKey()

# save output
cv2.imwrite("gender_detection.jpg", image)

# release resources
cv2.destroyAllWindows()
