# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
import ntpath
import sys
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True,
    help="Path to template image")
ap.add_argument("-i", "--images", required=True,
    help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
    help="Flag indicating whether or not to visualize each iteration")
ap.add_argument("-m", "--mirrored",
    help="Matches on template and mirrored version of template")
args = vars(ap.parse_args())
 
# load the image image, convert it to grayscale, and detect edges
template_org = cv2.imread(args["template"])
template_org = cv2.cvtColor(template_org, cv2.COLOR_BGR2GRAY)
template_org = cv2.Canny(template_org, 50, 200)
(tH, tW) = template_org.shape[:2]

templates = [ template_org ]
#Adds flipped version of template if found in arguments
if args.get("mirrored", False):
    template_flip = cv2.flip(template_org, 1) #Template flipped on the y axis
    templates.append(template_flip)


# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.jpg"):
    image = cv2.imread(imagePath)
    (iH, iW) = image.shape[:2]

    if tH >= iH and tW >= iW: #Skip image if smaller than template
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
 
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        for template in templates:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
    
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    
            # check to see if the iteration should be visualized
            if args.get("visualize", False):
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                    (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)
    
            # if we have found a new maximum correlation value, then ipdate
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
 
    # unpack the bookkeeping varaible and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    
    paddingW = int(tW * r * 0.1)
    paddingH = int(tH * r * 0.1)
    # print(startX, endX, startY, endY)
    startX -= paddingW
    startY -= paddingH
    endX += paddingW
    endY += paddingH

    display_image = image.copy()

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    
    keyCode = cv2.waitKey(0)
    if keyCode == 13: #Enter
        img_crop = image[startY-paddingH : endY + paddingH, startX-paddingW : endX + paddingW]
        writePath = './preprocessed/' + ntpath.basename(imagePath)
        cv2.imwrite(writePath, img_crop)
    elif keyCode == 27: #Esc
        cv2.destroyAllWindows()
        sys.exit()

cv2.destroyAllWindows()