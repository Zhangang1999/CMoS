import io

import cv2
import imageio
import numpy as np
from PIL import Image


def imageio_imread(fp):
    with open(fp, mode='rb') as f:
        img = Image.open(io.BytesIO(f))
    img = np.array(img)
    return img

def cv2_imread(fp):
    img = cv2.imread(fp)
    return img
