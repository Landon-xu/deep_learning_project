import cv2
from Find_Rect import find_rectangle


rect_images = find_rectangle('./image_license_plate/123.jpeg')
print(type(rect_images[0]))

# for image in rect_images:
#     cv2.imshow('rect_image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
