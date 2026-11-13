# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

import os

import cv2
import cvlib as cv
import numpy as np
from keras.models import load_model
from keras.utils import get_file
from tensorflow.keras.utils import img_to_array

CLASSES = ("man", "woman")

MODEL_FNAME = "gender_detection.model"
MODEL_DLOAD_LINK = (
    f"https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/{MODEL_FNAME}"
)
CACHE_SUBDIR = "pre-trained"

MODEL_PATH = get_file(
    MODEL_FNAME,
    MODEL_DLOAD_LINK,
    cache_subdir=CACHE_SUBDIR,
    cache_dir=os.getcwd(),
)


class GenderDetection:
    def __init__(self, model_path: str = MODEL_PATH):

        # load the model
        self.model = load_model(model_path)

    def detect_gender(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:

        # detect faces in the image
        face, _ = cv.detect_face(image)

        # loop through detected faces
        for idx, f in enumerate(face):
            # get corner points of face rectangle
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # crop the detected face region
            face_crop = np.copy(image[startY:endY, startX:endX])

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = self.model.predict(face_crop)[0]
            print(conf)
            print(CLASSES)

            # get label with max accuracy
            idx = np.argmax(conf)
            label = CLASSES[idx]

            label = f"{label}: {conf[idx] * 100:.2f}%"

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(
                image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        return image
