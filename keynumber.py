import cv2

cv2.imshow("hello", cv2.Canny(cv2.imread("template.jpg"), 1, 50))
while True:
    keyNum = cv2.waitKey(0)
    if keyNum == 27:
        break
    print(keyNum)