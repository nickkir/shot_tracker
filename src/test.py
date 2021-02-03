from imageprocessing import binary_image_processing as bip
from algorithms import rimfinder as rf
import matplotlib.pyplot as plt
import numpy as np
import cv2


image_path = "C:\\Users\\Nicholas\\Desktop\\basketball_detection_tests\\rims\\rim4.jpg"
rim = cv2.imread(image_path)
rim_rgb = cv2.cvtColor(rim, cv2.COLOR_BGR2RGB)
rim_rgb_2 = cv2.cvtColor(rim, cv2.COLOR_BGR2RGB)
rim_rgb_3 = cv2.cvtColor(rim, cv2.COLOR_BGR2RGB)
rim_hsv = cv2.cvtColor(rim_rgb, cv2.COLOR_RGB2HSV)

current_color = (7, 255, 237)

noisy_mask = bip.get_mask(rim_hsv, current_color)
blurred_mask = bip.apply_median_blur(noisy_mask, passes=3)
canny_mask = bip.apply_canny(blurred_mask)


fig = plt.figure()

test = rf.RimFinder(canny_mask)


true_ellipse = test.find_rim()
cv2.ellipse(rim_rgb, true_ellipse, (0,255,0))
plt.imshow(rim_rgb)

ellipse = test.get_inner_ellipse()

cv2.ellipse(rim_rgb_2, ellipse, (0,255,0))

plt.show()


