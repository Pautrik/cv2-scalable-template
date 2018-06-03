import cv2
import numpy as np
import imutils
import glob
import sys

img_rgb = cv2.imread('img.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
template_canny = cv2.Canny(template, 200, 300)
(tH, tW) = template.shape[:2]

while True:
    btnCode = cv2.waitKey(0)
    print(btnCode)

    if btnCode == 27: #Esc
        cv2.destroyAllWindows()
        sys.exit()
    elif btnCode == 13: #Enter
        break

cv2.destroyAllWindows()
sys.exit()
