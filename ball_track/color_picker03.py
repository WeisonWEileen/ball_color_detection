#---------------------------#
# 本demo交互式识别HSV颜色空间的三个通道的最大值和最小值,通过w键和s键能够扩大和缩小采样圆大小
# 按q退出
#---------------------------#

import cv2
import numpy as np
from imutils.video import VideoStream
import imutils
import time

# 初始化圆的参数
radius = 20  # 圆的半径
circle_center = (320, 240)  # 圆的初始中心位置
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
    global circle_center, mouse_pressed, radius
    if event == cv2.EVENT_LBUTTONDOWN:
        circle_center = (x, y)
        mouse_pressed = True
    elif event == cv2.EVENT_MOUSEMOVE and mouse_pressed:
        circle_center = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        update_hsv_values(circle_center, radius, param)

# 开始捕捉视频
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # 让摄像头预热

# 创建窗口并设置鼠标回调函数
cv2.namedWindow("Color Picker")

while True:
    # 从视频流中读取帧
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 更新鼠标回调函数和显示圆
    cv2.setMouseCallback("Color Picker", mouse_callback, hsv_img)
    img_copy = frame.copy()
    cv2.circle(img_copy, circle_center, radius, (0, 255, 0), 2)
    
    # 如果鼠标被按下，更新HSV值
    if mouse_pressed:
        update_hsv_values(circle_center, radius, hsv_img)

    # 显示图像
    cv2.imshow("Color Picker", img_copy)

    # 打印HSV值
    print(f"Upper HSV: {upper} Lower HSV: {lower}")

    # 捕获键盘事件
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        radius += 1
    elif key == ord('s'):
        radius -= 1
    radius = max(1, radius)  # 确保半径至少为1,防止太小无法看清
    
    # 按下 'q' 键退出循环
    if key == ord('q'):
        break

# 释放资源
cv2.destroyAllWindows()
vs.stop()
