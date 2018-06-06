import cv2
import numpy as np
import imutils
from glob import glob
import sys

imgPath = "."


template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
template_canny = cv2.Canny(template, 200, 300)
(tH, tW) = template.shape[:2]


files = glob(imgPath + "/*.jpg")
files += glob(imgPath + "/.*jpeg")
files += glob(imgPath + "/.*png")

fileIndex = 0
while fileIndex < len(files):
    img_rgb = cv2.imread(files[fileIndex])
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    found = None
    scale = 0.5

    while True:
        btnCode = cv2.waitKey(0)
        print(btnCode)

        if btnCode == 27:  # Esc
            cv2.destroyAllWindows()
            sys.exit()
        elif btnCode == 13:  # Enter
            break
        elif btnCode == 82:  # Up
            pass
        elif btnCode == 84:  # Down
            pass
        elif btnCode == 81:  # Left
            pass
        elif btnCode == 83:  # Right
            pass
        elif btnCode == 255:  # Delete
            pass
        

cv2.destroyAllWindows()
sys.exit()
