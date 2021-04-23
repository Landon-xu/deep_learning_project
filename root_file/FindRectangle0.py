import cv2
import numpy as np
from PIL import Image
from findplate.testnetwork import detect
from findplate.testnetwork import identify


# preprocess the image
def preprocess(img):
    # 将图片转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 车牌照为蓝色，设置蓝色的hsv阈值，提取出图片中的蓝色区域
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    plate_color_img = (((h > 100) & (h < 124))) & (s > 120) & (v > 60)
    # 将图片数据格式转为8UC1的二值图
    plate_color_img = plate_color_img.astype('uint8') * 255
    # 对图片进行膨胀处理，使车牌成为一个整体
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    plate_color_img = cv2.dilate(plate_color_img, element, iterations=1)
    return plate_color_img


# 找到车牌位置
def findPlate(plate_color_img, im):
    # 在膨胀后的二值图像中寻找所有的轮廓，并存入数组
    contours, hierarchy = cv2.findContours(plate_color_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    # 遍历轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        # 去除面积很小的轮廓
        if (area < (1 / 500 * plate_color_img.shape[0] * plate_color_img.shape[1])):
            continue

        # 获取轮廓的最小外接矩形
        rect = cv2.minAreaRect(contour)
        rect_point = cv2.boxPoints(rect)
        rect_point = np.int0(rect_point)

        cv2.drawContours(im, [rect_point], 0, (0, 0, 255), 2)
        cv2.imwrite('contours.png', im)


def recognition(path):
    im = cv2.imread(path)
    height, width = im.shape[:2]  # get height and width of the image
    plate_color_img = preprocess(im)
    # cv_show("plate_color_img", plate_color_img)
    findPlate(plate_color_img, im)


def cv_show(img, name):
    cv2.imshow(img, name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_line(img, ptStart, ptEnd):
    point_color = (0, 0, 255)  # BGR: red
    thickness = 1
    lineType = 8
    cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)


if __name__ == "__main__":
    # path = input('Please input path:')
    path = './image/Chinese_plate.jpeg'
    recognition(path)