import cv2


# Given an HSV color and an HSV image, creates and appropriate mask (H: 0-179, S: 0-255, V: 0-255)
# TODO color range is very arbitrary, look into making it more scientific
def get_mask(hsv_image, hsv_color, hue_range=7):
    (h, s, v) = hsv_color

    lower_h = min([0, h-hue_range])
    lower_s = 100
    lower_v = 100

    upper_h = min([179, h+hue_range])
    upper_s = 255
    upper_v = 255

    lower_bound = (lower_h, lower_s, lower_v)
    upper_bound = (upper_h, upper_s, upper_v)

    output_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    return output_mask


# Gets a new image that has been median blurred (i.e. mode blurred) the specified number of times
def apply_median_blur(bin_img, passes=1, kernel_size=5):
    copy = bin_img

    for i in range(passes):
        copy = cv2.medianBlur(copy, kernel_size)

    return copy


# Gets an image of the edges found with Canny algorithm
def apply_canny(bin_img, min_size=50, max_size=50):
    edges = cv2.Canny(bin_img, 200, 200)

    return edges
