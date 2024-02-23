#---------------------------#
#本demo交互式识别图片的HSV颜色空间的三个通道的最大值和最小值,通过w键和s键能够扩大和缩小采样圆大小
#按q退出
#---------------------------#

import cv2
import numpy as np
from collections import deque
from imutils.video import VideoStream
import imutils
import time
# 初始化圆的参数
radius = 20  # 圆的半径
circle_center = (0, 0)  # 圆的初始中心位置
mouse_pressed = False  # 鼠标是否被按下

# HSV颜色空间的最大值和最小值
upper = [0, 0, 0]
lower = [255, 255, 255]

# 更新HSV最大值和最小值的函数
def update_hsv_values(circle_center, radius, hsv_img):
    global upper, lower
    x, y = circle_center
    # 制作一个和图像大小一样的掩膜，其中圆内的区域为1，其他区域为0
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, (255), thickness=-1)
    
    # 应用掩膜获取圆内的像素点
    masked_hsv = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    
    # 转换masked_hsv为数组，然后去除所有黑色的点（值为0）
    masked_hsv_array = masked_hsv.reshape(-1, 3)
    masked_hsv_array = masked_hsv_array[np.all(masked_hsv_array != [0, 0, 0], axis=1)]
    
    # 更新最大值和最小值
    if masked_hsv_array.size > 0:
        lower = np.min(masked_hsv_array, axis=0).tolist()
        upper = np.max(masked_hsv_array, axis=0).tolist()

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global circle_center, radius
    hsv_img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        circle_center = (x, y)
        update_hsv_values(circle_center, radius, hsv_img)
    elif event == cv2.EVENT_MOUSEMOVE:
        circle_center = (x, y)
        update_hsv_values(circle_center, radius, hsv_img)
    print(f"Upper HSV: {upper} Lower HSV: {lower}")
# 读取图像并转换到HSV颜色空间
path = "./pict/test.png"
img = cv2.imread(path)
if img is None:
    raise ValueError("Image not found for this path")

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 创建窗口并设置鼠标回调函数
cv2.namedWindow("Color Picker")
cv2.setMouseCallback("Color Picker", mouse_callback, hsv_img)




while True:
    



    img_copy = img.copy()
    cv2.circle(img_copy, circle_center, radius, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow("Color Picker", img_copy)

    # 捕获键盘事件
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        radius += 1
    elif key == ord('s'):
        radius -= 1
    radius = max(1, radius)
    
    # 按下 'q' 键退出循环
    if key == ord('q'):
        break

cv2.destroyAllWindows()
