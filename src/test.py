from imageprocessing import binary_image_processing as bip
from algorithms import rimfinder as rf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def test_rim_finding(image_path, color):
    rim = cv2.imread(image_path)
    rim_rgb = cv2.cvtColor(rim, cv2.COLOR_BGR2RGB)
    rim_hsv = cv2.cvtColor(rim_rgb, cv2.COLOR_RGB2HSV)

    noisy_mask = bip.get_mask(rim_hsv, color)
    blurred_mask = bip.apply_median_blur(noisy_mask, passes=3)
    canny_mask = bip.apply_canny(blurred_mask)

    rim_finding_obj = rf.RimFinder(canny_mask)

    ellipse = rim_finding_obj.find_rim()

    cv2.ellipse(rim_rgb, ellipse, (0, 255, 0))
    plt.imshow(rim_rgb)
    plt.show()


