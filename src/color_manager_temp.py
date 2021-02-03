import cv2
import numpy as np
from PIL import Image


def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:  # checks mouse left button down condition
        colors = array_image[y,x]

        print(colors)
        return colors


image_path = "C:\\Users\\Nicholas\\Desktop\\basketball_detection_tests\\rims\\rim3.jpg"

rim = cv2.imread(image_path)


image = rim

pil_image = Image.fromarray(image)
pil_image = pil_image.resize((400, 400))

array_image = np.array(pil_image)




def show_image(im_arr):
    cv2.namedWindow('Rim Finding Example')
    cv2.setMouseCallback('Rim Finding Example', mouseRGB)
    while(1):
        cv2.imshow('Rim Finding Example', im_arr)
        if cv2.waitKey(20) & 0xFF == 27:
            break

show_image(array_image)
