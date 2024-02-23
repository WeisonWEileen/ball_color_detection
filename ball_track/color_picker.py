#此文件可以通过连续鼠标点击获取某张图片的HSV的各个通道的最大值和最小值
import cv2
upper = [0,0, 0]
lower = [255,255,255]

#初始化交互圆的参数
radius = 20
circle_center=(0,0)
mouse_press = False

# 定义鼠标交互函数
def mouseColor(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(str.upper(out), color[y, x])  #输出图像坐标(x,y)处的HSV的值
        if (color[y,x][0]<lower[0]):
            lower[0]=color[y,x][0]
        if (color[y,x][1]<lower[1]):
            lower[1]=color[y,x][1]
        if (color[y,x][2]<lower[2]):
            lower[2]=color[y,x][2]
        if (color[y,x][0]>upper[0]):
            upper[0]=color[y,x][0]
        if (color[y,x][1]>upper[1]):
            upper[1]=color[y,x][1]
        if (color[y,x][2]>upper[2]):
            upper[2]=color[y,x][2]
        print(f"uppper {upper} lower {lower}")
        

path, out = "./pict/test.png hsv".split()
print("The path is ",path)
img = cv2.imread(path)  #读进来是BGR格式

if img is None:
    raise ValueError("Image not found for this path ")
# 进行颜色格式的转换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #变成灰度图
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  #变成HSV格式
if out == 'bgr':
    color = img
if out == 'gray':
    color = gray
if out == 'hsv':
    color = hsv
cv2.namedWindow("Color Picker")
cv2.setMouseCallback("Color Picker", mouseColor)
cv2.imshow("Color Picker", img)
if cv2.waitKey(0):
    cv2.destroyAllWindows()
