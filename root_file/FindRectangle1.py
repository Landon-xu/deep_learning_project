import cv2
import numpy as np

'''仅做矩形检测'''

def crop_rect(img, rect):  # 旋转图像转正 https://blog.csdn.net/loovelj/article/details/90080725
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)  # 利用getRotationMatrix2D实现旋转：getRotationMatrix2D(center, angle, scale)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot


im = cv2.imread('./image/123.jpeg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) # 转换为灰度图

ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 大津阈值
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL 定义只检测外围轮廓

cnts = contours[0]

for cnt in cnts:

    # 最小外接矩形框，有方向角
    rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

    # 过滤不合格的矩形
    width = rect[1][0]
    height = rect[1][1]
    if(width == 0 or height == 0):
        continue
    if(width / height < 0.2):
        continue
    if((height < 0.05 * im.shape[0]) or (width < 0.05 * im.shape[1])):
        continue
    if((height > 0.8 * im.shape[0]) or (width > 0.8 * im.shape[1])):
        continue
    print(f'width: {rect[1][0]}\t', end='')
    print(f'height: {rect[1][1]}')
    img_crop, img_rot = crop_rect(im, rect)
    cv2.imshow('b', img_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
    box = np.int0(box)
    cv2.drawContours(im, [box], 0, (0, 0, 255), 2)


cv2.imshow('a', im)
# cv2.imwrite('./result.jpg',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
