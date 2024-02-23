import cv2
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    raise IOError("无法打开摄像头")

# 逐帧读取视频
while True:
    # 读取一帧
    ret, frame = cap.read()
    cv2.imshow('origin Frame', frame)


    # 如果正确读取帧，ret为True
    if not ret:
        print("无法读取摄像头帧，退出...")
        break

    # 对于每个像素，比较蓝色通道的值与红色和绿色通道的值
    #好像这里写错了？这里并没有过滤掉不是蓝色最大的值
    max_blue = np.maximum(np.maximum(frame[:, :, 0], frame[:, :, 1]), frame[:, :, 2])
    # 创建一个只包含蓝色通道最大值的掩码
    blue_mask = (frame[:, :, 0] == max_blue).astype(np.uint8) * 255

    cv2.imshow('blue_mask', blue_mask)

    # 创建新的图像，如果蓝色通道不是最大的，就设置所有通道为0
    masked_image = cv2.bitwise_and(frame, frame, mask=blue_mask)
	
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)#转换为HSV空间


    #定义
    blueLower = (104, 57, 122)
    blueLower = (118, 255, 255)
    # 将图像转换到HSV颜色空间
    mask_HSV = cv2.inRange(masked_image, blueLower, blueLower)
    

    # 显示处理后的帧
    cv2.imshow('Processed Frame', masked_image)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
