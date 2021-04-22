import cv2 as cv
import numpy as np

'''
加了边缘检测，膨胀腐蚀处理
目前效果不如FindRectangle1好
'''

# Canny边缘检测
def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    # canny_output = cv.Canny(image, 50, 150)  #?
    cv.imshow("canny_output", canny_output)
    cv.imwrite("canny_output.png", canny_output)
    return canny_output


src = cv.imread("./image/777.jpeg")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

'''?'''
# grayimg = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # RGB转灰度图
# # 进行开运算，用来去除噪声
# kernel = np.ones((5,5),np.uint8)
# opening = cv.morphologyEx(grayimg, cv.MORPH_OPEN, kernel)
'''?'''

binary = canny_demo(src)  # 进行边缘检测，换成二值图像
# binary = canny_demo(opening)  # 进行边缘检测，换成二值图像?
# closing = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)  # 闭运算？
k = np.ones((3, 3), dtype=np.uint8)
# binary = cv.morphologyEx(closing, cv.MORPH_DILATE, k)  # 膨胀处理？
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)  # 膨胀处理

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    area = cv.contourArea(contours[c])  # 轮廓的面积
    arclen = cv.arcLength(contours[c], True)  # 轮廓的周长

    # 根据面积和周长过滤不合格的轮廓
    if (area < (0.0001 * src.shape[0] * src.shape[1]) or arclen < 0.001 * src.shape[0]):
        continue
    rect = cv.minAreaRect(contours[c])  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

    # 过滤不合格的矩形
    width = rect[1][0]
    height = rect[1][1]
    # print(f'width: {rect[1][0]}\t', end='')
    # print(f'height: {rect[1][1]}')
    if (width == 0 or height == 0):
        continue
    if (width / height < 0.2):
        continue
    if ((height < 0.005 * src.shape[0]) or (width < 0.005 * src.shape[1])):
        continue

    cx, cy = rect[0]
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(src, [box], 0, (0, 0, 255), 2)
    cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)


# 显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()
