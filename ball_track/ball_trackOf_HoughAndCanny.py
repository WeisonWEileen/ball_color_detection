import numpy as np
import cv2 as cv

img = cv.imread('./pict/bi.png', cv.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
if circles is not None:
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        # Draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
else:
    raise ValueError("No circles could be detected. Please check the image content or adjust the detection parameters.")

cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()