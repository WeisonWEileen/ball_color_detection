import cv2
src = cv2.imread("../41F195BC0547C4B8D6BDF83EF310E8EB.png")

# 图像预处理
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 7)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 60, param1=190, param2=30, minRadius=50, maxRadius=0)

# TypeError: Argument 'radius' is required to be an integer
for x, y, r in circles[0]: cv2.circle(src, (int(x), int(y)), int(r), (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow('circle', src)
cv2.waitKey(0)

cv2.destroyWindow()
