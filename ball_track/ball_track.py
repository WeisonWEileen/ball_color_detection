# 这是一个基于HSV颜色空间识别的一个demo，但是目前对于
# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())


# 全局交互的HSV阈值调整
#233蓝色球阈值
yellowLower = (104, 57, 122)
yellowUpper = (118, 255, 255)
#小黑屋蓝色球阈值
# yellowLower = (106, 57, 122)
# yellowUpper = (113, 255, 198)
#全句交互霍夫圆参数
minDist = 20
param1_ = 50
param2_ = 30

#标记两个窗口
cv2.namedWindow('Frame')
cv2.namedWindow('HSV_Binary+Canny+Hough')
def nothing(x):
    pass

#toolbar 的创建
# Create trackbars for HSV lower bounds and upper bounds
cv2.createTrackbar('Lower H', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Lower S', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Lower V', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Upper H', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Upper S', 'Frame', 0, 255, nothing)
cv2.createTrackbar('Upper V', 'Frame', 0, 255, nothing)
# Set initial positions of trackbars
cv2.setTrackbarPos('Lower H', 'Frame', yellowLower[0])
cv2.setTrackbarPos('Lower S', 'Frame', yellowLower[1])
cv2.setTrackbarPos('Lower V', 'Frame', yellowLower[2])
cv2.setTrackbarPos('Upper H', 'Frame', yellowUpper[0])
cv2.setTrackbarPos('Upper S', 'Frame', yellowUpper[1])
cv2.setTrackbarPos('Upper V', 'Frame', yellowUpper[2])


cv2.createTrackbar('minDist', 'HSV_Binary+Canny+Hough', 1, 255, nothing)
cv2.createTrackbar('param1_', 'HSV_Binary+Canny+Hough', 100, 600, nothing)
cv2.createTrackbar('param2_', 'HSV_Binary+Canny+Hough', 20, 100, nothing)
cv2.setTrackbarPos('minDist', 'HSV_Binary+Canny+Hough', minDist)
cv2.setTrackbarPos('param1_', 'HSV_Binary+Canny+Hough', param1_)
cv2.setTrackbarPos('param2_', 'HSV_Binary+Canny+Hough', param2_)





# yellowLower = (30,39,96)
# yellowUpper = (59,70,100)
pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)

def B_R(frame):
    # 计算每个像素BGR值中最大的值
    max_blue = np.maximum.reduce([frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]])
    # 创建蓝色掩码，只有蓝色值是最大值的像素才被保留
    blue_mask = (frame[:, :, 0] == max_blue).astype(np.uint8) * 255
    # 显示蓝色掩码
    cv2.imshow('blue_mask', blue_mask)
    # 应用掩码到图像
    masked_image = cv2.bitwise_and(frame, frame, mask=blue_mask)
    return masked_image

def gray_to_hough(frame):
	#原图转灰度再转canny
	print('Image dimensions of gray:', frame.shape)
	frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	frame02 = cv2.medianBlur(frame,5)
	frame03 = cv2.Canny(frame02,50,300)
	circles = cv2.HoughCircles(frame03,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(frame03,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		cv2.circle(frame03,(i[0],i[1]),2,(0,0,255),3)
	return frame03

#画出霍夫圆检测的结果:圆+一个外切正方形表示圆的位置
# def draw_rect_for_hough(frhsvame,circles):
# 	for i in circles[0, :]:
# 			# draw the outer circle
# 			cv2.circle(frame, (i[0], i[1]), i[2], (255, 255, 0), 2)
# 			# draw the center of the circle
# 			cv2.circle(frame, (i[0], i[1]), 2, (255, 255, 0), 3)

# 			# 获取圆的半径和中心坐标
# 			radius = i[2]
# 			center = (i[0], i[1])
# 			votes = i[3]
			
# 			# 计算外切正方形的左上角和右下角坐标
# 			square_size = 2 * radius
# 			x1 = center[0] - radius
# 			y1 = center[1] - radius
# 			x2 = x1 + square_size
# 			y2 = y1 + square_size
			
# 			# 绘制外切正方形
# 			cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  
# 			# 上面使用蓝色绘制正方形似乎没有效果?

# 			   # Display the vote count   
# 	print(f"circle01{circles[0]}")

def draw_rect_for_hough(frame,circles):
	# print(circles)
	if len(circles) > 0 :
		i = circles[0,0]
		# draw the outer circle
		cv2.circle(frame, (i[0], i[1]), i[2], (255, 255, 0), 2)
		# draw the center of the circle
		cv2.circle(frame, (i[0], i[1]), 2, (255, 255, 0), 3)

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
		cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  
		# 上面使用蓝色绘制正方形似乎没有效果?

				# Display the vote count   
		print(f"circle01{circles[0]}")



while True:

	#更新全局交互的变量
	yellowLower = (cv2.getTrackbarPos('Lower H', 'Frame'),cv2.getTrackbarPos('Lower S', 'Frame'),cv2.getTrackbarPos('Lower V', 'Frame'))
	yellowUpper = (cv2.getTrackbarPos('Upper H', 'Frame'),cv2.getTrackbarPos('Upper S', 'Frame'),cv2.getTrackbarPos('Upper V', 'Frame'))
	minDist = (cv2.getTrackbarPos('minDist', 'HSV_Binary+Canny+Hough'))
	param1_ = (cv2.getTrackbarPos('param1_', 'HSV_Binary+Canny+Hough'))
	param2_ = (cv2.getTrackbarPos('param2_', 'HSV_Binary+Canny+Hough'))
	
	# grab the current frame
	frame = vs.read()
	frame = B_R(frame)
	


	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# resize the frame, blur it, and convert it to the HSV

	#灰度图像的hough圆检测
	# edges02 = gray_to_hough(frame)
	# cv2.imshow("Gray_to_Hough", edges02)


	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)#高斯模糊，抑制摄像头噪声
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)#转换为HSV空间
	
	


	#高斯模糊后的原图canny边缘检测
	# edges_0 = cv2.Canny(hsv, 50, 300)
	#常识这边显示霍夫圆检测的结果
	# circles_0 = cv2.HoughCircles(edges_0, cv2.HOUGH_GRADIENT, 1, 60, param1=param1_, param2=param2_, minRadius=0, maxRadius=0)
	# cv2.imshow('direct_canny', edges_0)
	# if circles_0 is not None:
	# 	num_circles_0 = circles_0.shape[1]
	# 	circles_0 = np.uint16(np.around(circles_0))
	# 	draw_rect_for_hough(edges_0,circles_0)

	# construct a mask for the preset color between lower and higher
	mask = cv2.inRange(hsv, yellowLower, yellowUpper)#对图像进行掩膜运算
	mask = cv2.erode(mask, None, iterations=5)
	mask = cv2.dilate(mask, None, iterations=5)


	#以下是HSV掩膜二值化后使用Canny边缘检测后用霍夫圆的方法
	#二值化图像后的canny边缘检测
	edges = cv2.Canny(mask, 50, 300)
	# Hough circle detection
	circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 60, param1=param1_, param2=param2_, minRadius=0, maxRadius=0)
	print(circles)

	if circles is not None:
		num_circles = circles.shape[1]
		print(f"{num_circles} detected in HSV binary image")
		circles = np.uint16(np.around(circles))
		draw_rect_for_hough(edges,circles)
		
	else :
		print(f"0 detected in HSV binary image")
		# print(f"0 detected in HSV binary image {num_circles_0}detected in direct canny image")

    # 显示二值化检测的结果
	cv2.imshow('HSV Binary Circles', mask)
 	# 显示霍夫圆检测的结果
	cv2.imshow('HSV_Binary+Canny+Hough', edges)
	# cv2.imshow("Frame02",mask)
    	# find contours in the mask and initialize the current
	# (x, y) center of the ball

	#---------------------------------------------#
	#以下是使用findContours的进行提取的方法
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)


		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)



	# update the points queue
	pts.appendleft(center)

    # loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera

else:
	vs.release()
# close all windows
cv2.destroyAllWindows()


