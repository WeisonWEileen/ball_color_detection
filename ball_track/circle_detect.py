import cv2
import math
import numpy as np

#--------------------------------------#
##----本demo是基于HSV色彩空间的三色球的颜色识别
##----需要注意，红色在HSV色彩空间的两端，因此它有两个区间
def color_identify(self, image, target_color='red'):
        _img = image.copy()

        _color_space = 'hsv'  # 'hsv' 或 'lab'

        # 获取图像宽高, 默认 640, 480
        img_h, img_w = _img.shape[:2]

        # 图像已经是 640*480 了
        # img_resize = cv2.resize(_img, SIZE, interpolation=cv2.INTER_NEAREST)

        # 高斯模糊, 抑制摄像头噪声(一些白点会被过滤掉，但影响不大, 因为他们的面积不会超过被识别物体?)
        imt_g_blur = cv2.GaussianBlur(_img, (7, 7), 0)  # 可以改成均值模糊

        frame_mask = None  # 掩膜结果
        if _color_space == 'hsv':
            hsv_data = {
                'red': {'lower': (0, 43, 46), 'upper': (10, 255, 255)},
                'red1': {'lower': (156, 43, 46), 'upper': (180, 255, 255)},
                'blue': {'lower': (100, 43, 46), 'upper': (124, 255, 255)},
                'green': {'lower': (35, 43, 46), 'upper': (77, 255, 255)},
            }

            # 将图像转换到 HSV 色彩空间
            frame_hsv = cv2.cvtColor(imt_g_blur, cv2.COLOR_BGR2HSV)

            # 对图像进行掩膜运算
            frame_mask = cv2.inRange(frame_hsv,
                                     hsv_data[target_color]['lower'],
                                     hsv_data[target_color]['upper'])

            # 红色有两个区间, 所以做与运算
            if target_color == 'red':
                frame_mask1 = cv2.inRange(frame_hsv,
                                          hsv_data['red1']['lower'],
                                          hsv_data['red1']['upper'])
                frame_mask = cv2.bitwise_or(frame_mask, frame_mask1)

        elif _color_space == 'lab':
            lab_data = {
                'red': {'min': (0, 43, 46), 'max': (10, 255, 255)},
                'red1': {'min': (156, 43, 46), 'max': (180, 255, 255)},
                'blue': {'min': (100, 43, 46), 'max': (124, 255, 255)}
            }

            # 将图像转换到LAB空间
            frame_lab = cv2.cvtColor(imt_g_blur, cv2.COLOR_BGR2LAB)

            # 对原图像进行掩模运算
            frame_mask = cv2.inRange(frame_lab,
                                     np.array(lab_data[target_color]['min']),
                                     np.array(lab_data[target_color]['max']))

        # 颜色空间设置错误
        if frame_mask is None:
            return -1, -1, -1, 0

        # 开运算, 消除白点
        # opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # 开运算, 消除小白点
        # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 闭运算, 闭合被识别的障碍物
        closed = frame_mask

        # debug
        # cv2.imshow('color_identify', closed)
        # cv2.waitKey(10)  # 等待 10ms

        # 找出所有外轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 如果未发现任何颜色, 或者说 没有发现轮廓
        if len(contours) == 0:
            return -1, -1, -1, 0

        # 对轮廓对象按面积进行排序, 降序, 面积最大的为序号为0
        c_sorted = sorted(contours,
                          key=lambda x:
                          math.fabs(
                              cv2.contourArea(x)
                          ), reverse=True)

        c_0 = c_sorted[0]   # 最大轮廓

        # 如果面积小于 100, 认为是噪声, 进行下一次处理
        if math.fabs(cv2.contourArea(c_0)) <= 100:
            return -1, -1, -1, 0

        return tuple(
            map(int, cv2.boundingRect(c_0))
        )