import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./pict/bi.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img, 50, 300)

##用于显示检测出来的圆
img_show = img.copy()

# Hough circle detection
circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 60, param1=300, param2=40, minRadius=0, maxRadius=0)

if circles is not None:
    num_circles = circles.shape[1]
    print(f"{num_circles} was dectected here")
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(img_show, (i[0], i[1]), i[2], (255, 255, 0), 2)
        # draw the center of the circle
        cv.circle(img_show, (i[0], i[1]), 2, (255, 255, 0), 3)

        # 获取圆的半径和中心坐标
        radius = i[2]
        center = (i[0], i[1])
        
        # 计算外切正方形的左上角和右下角坐标
        square_size = 2 * radius
        x1 = center[0] - radius
        y1 = center[1] - radius
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        # 绘制外切正方形
        cv.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 使用蓝色绘制正方形


    while True:

        plt.subplot(131),plt.imshow(img,cmap = 'gray')
        plt.title('Original Binary Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(323),plt.imshow(img_show,cmap = 'gray')
        plt.subplot(133),plt.imshow(cv.cvtColor(img_show, cv.COLOR_BGR2RGB))
        plt.title('Hough Circle Detect'), plt.xticks([]), plt.yticks([])
        plt.show()

        # cv.imshow('detected circles', img)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):  # 检测按下的键是否是 'q'
            break

    cv.destroyAllWindows()

else:
    print("No circles were detected.")

