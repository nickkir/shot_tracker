import numpy as np
from typing import List
import cv2
import math


# Class for storing all the pixels that delimit a new region in a binary image
class AugmentedBinaryImageStrip:
    def __init__(self, quintuples_list) -> None:
        assert len(quintuples_list) > 0
        # Make it quicker to access length
        self.length = len(quintuples_list)

        self.pixel_array = np.empty(self.length, dtype=np.dtype([('x', np.int32), ('y', np.int32), ('value', np.uint8),
                                                                 ('line', np.int32), ('column', np.int32)]))
        for i in range(self.length):
            self.pixel_array[i] = quintuples_list[i]

        # The following attributes will be used a lot, so they are stored as attributes
        # Calling the getter will initialize their value
        self.delimiting_pixels = AugmentedBinaryImageStrip.__compute_ordered_delimiting_pixels_arr(self.pixel_array)
        self.regions = AugmentedBinaryImageStrip.__compute_regions(self.delimiting_pixels)
        self.amount_edge_intersections = np.count_nonzero(self.regions)

    # helper function for converting pixel lists into pixel arrays
    @staticmethod
    def __convert_list_quintuples_into_array(pixel_list):
        number_of_pixels = len(pixel_list)
        pixels_arr = np.empty(number_of_pixels, dtype=np.dtype([('x', np.int32), ('y', np.int32), ('value', np.uint8),
                                                                ('line', np.int32), ('column', np.int32)]))

        for i in range(number_of_pixels):
            pixels_arr[i] = pixel_list[i]

        return pixels_arr

    # static helper method used by the constructor to get all the delimiting pixels
    # pixels are in ordering encountered, from right to left / top to bottom
    @staticmethod
    def __compute_ordered_delimiting_pixels_arr(pixel_arr):
        # first pixel obviously is start of new region
        delimiting_pixels_list = [pixel_arr[0]]

        for i in range(1, len(pixel_arr)):
            current_pixel = pixel_arr[i]
            if current_pixel['value'] != delimiting_pixels_list[-1]['value']:  # we have changed from black to white
                delimiting_pixels_list.append(current_pixel)

        delimiting_pixels_arr = AugmentedBinaryImageStrip.__convert_list_quintuples_into_array(delimiting_pixels_list)

        return delimiting_pixels_arr

    # static method to compute the number of intersections
    @staticmethod
    def __compute_regions(delimiting_pixel_arr):
        regions_list = list(map(lambda x: x['value'], delimiting_pixel_arr))
        regions_arr = np.array(regions_list)

        return regions_arr

    # returns list of quituples (x, y, value) representing the pixels of the first and last edge intersection in order
    def get_outer_intersections_list(self):
        ordered_intersections_arr = self.delimiting_pixels
        backwards_intersections_arr = np.flip(ordered_intersections_arr)

        outer_intersections = []
        first_edge_pixel = None

        # finds and appends the first white delimiting pixel
        for pixel_quintuple in ordered_intersections_arr:
            if pixel_quintuple['value'] == 255:
                outer_intersections.append(pixel_quintuple)
                first_edge_pixel = pixel_quintuple
                break

        # finds and appends the last white delimiting pixel
        for pixel_quintuple in backwards_intersections_arr:
            if pixel_quintuple['value'] == 255 and pixel_quintuple != first_edge_pixel:
                outer_intersections.append(pixel_quintuple)
                break

        return outer_intersections

    def get_inner_intersections_list(self):
        all_white_pixels = []

        for pixel_quintuple in self.delimiting_pixels:
            if pixel_quintuple['value'] == 255:
                all_white_pixels.append(pixel_quintuple)

        if len(all_white_pixels) <= 2:
            return []
        else:
            all_white_pixels.pop(0)
            all_white_pixels.pop(-1)
            return all_white_pixels

    def get_ordered_white_pixels_list(self):
        white_pixel_list = []

        for pixel_quintuple in self.delimiting_pixels:
            if pixel_quintuple['value'] == 255:
                white_pixel_list.append(pixel_quintuple)

        return white_pixel_list


class AugmentedBinaryImage:
    def __init__(self, binary_img):
        aug_rows = AugmentedBinaryImage.__create_augmented_rows(binary_img)
        aug_cols = AugmentedBinaryImage.__create_augmented_cols(binary_img)
        self.augmented_rows_list = aug_rows
        self.augmented_columns_list = aug_cols
        self.height = len(aug_rows)
        self.width = len(aug_cols)

        # The following are gonna be needed a lot during the object's life cycle, so instead of computing it once
        # (which is pretty long), we check if the object attributes have been initialized before calculating
        # they are private to force the client to access through the getters
        self.row_intersections_bool_arr = \
            AugmentedBinaryImage.__compute_4_intersections_bool_arr(self.augmented_rows_list)
        self.column_intersections_bool_arr = \
            AugmentedBinaryImage.__compute_4_intersections_bool_arr(self.augmented_columns_list)
        self.longest_4_intersections_rows_series_range = \
            AugmentedBinaryImage.__get_longest_intersections_range(self.row_intersections_bool_arr)
        self.longest_4_intersections_columns_series_range = \
            AugmentedBinaryImage.__get_longest_intersections_range(self.column_intersections_bool_arr)

    # static helper method for creating the list of augmented rows
    @staticmethod
    def __create_augmented_rows(bin_img):
        num_rows = bin_img.shape[0]
        num_cols = bin_img.shape[1]

        augmented_rows_list = []

        # Some fancy index fenangaling happens here to go from (line, column) -> (x, y)
        for i in range(num_rows):
            quintuples_list = []
            for j in range(num_cols):
                quintuples_list.append((j+1, num_rows-i, bin_img[i][j], i, j))
            augmented_rows_list.append(AugmentedBinaryImageStrip(quintuples_list))

        return augmented_rows_list

    # static helper method for creating the list of augmented columns
    @staticmethod
    def __create_augmented_cols(bin_img):
        num_rows = bin_img.shape[0]
        num_cols = bin_img.shape[1]

        augmented_cols_list = []

        for i in range(num_cols):
            quintuples_list = []
            for j in range(num_rows):
                quintuples_list.append((i+1, num_rows-j, bin_img[j][i], j, i))
            augmented_cols_list.append(AugmentedBinaryImageStrip(quintuples_list))

        return augmented_cols_list

    @staticmethod
    def __compute_4_intersections_bool_arr(augmented_strip_list):
        num_rows = len(augmented_strip_list)
        bool_arr = np.empty(num_rows)

        for i in range(num_rows):
            if augmented_strip_list[i].amount_edge_intersections == 4:
                bool_arr[i] = True
            else:
                bool_arr[i] = False

        return bool_arr

    @staticmethod
    def __get_longest_intersections_range(bool_arr):
        largest_intersect_range = AugmentedBinaryImage.__get_longest_true_series_range(bool_arr)

        return largest_intersect_range

    # returns the range (inclusively) of the longest series of 1s (Trues) in a boolean array
    # Returns the first instance in case of ties
    @staticmethod
    def __get_longest_true_series_range(boolean_arr):
        assert len(boolean_arr) > 0

        max_value = boolean_arr[0]
        max_index = 0
        for i in range(1, len(boolean_arr)):
            if boolean_arr[i] == 0:
                continue
            else:
                boolean_arr[i] = boolean_arr[i-1] + 1

            if boolean_arr[i] > max_value:
                max_value = boolean_arr[i]
                max_index = i

        return int(max_index-max_value+1), max_index

    def get_inner_rim_points_quintuples_arr(self):
        inner_points_list = []

        row_start, row_end = self.longest_4_intersections_rows_series_range
        col_start, col_end = self.longest_4_intersections_columns_series_range

        for i in range(row_start, row_end+1):
            inner_points_list += self.augmented_rows_list[i].get_inner_intersections_list()

        for i in range(col_start, col_end+1):
            inner_points_list += self.augmented_columns_list[i].get_inner_intersections_list()

        inner_points_arr = np.array(inner_points_list)
        np.unique(inner_points_arr)

        return inner_points_arr

    def get_outer_rim_points_quintuples_arr(self):
        outer_points_list = []

        row_start, row_end = self.longest_4_intersections_rows_series_range
        col_start, col_end = self.longest_4_intersections_columns_series_range

        for i in range(row_start, row_end + 1):
            outer_points_list += self.augmented_rows_list[i].get_outer_intersections_list()

        for i in range(col_start, col_end + 1):
            outer_points_list += self.augmented_columns_list[i].get_outer_intersections_list()

        outer_points_arr = np.array(outer_points_list)
        np.unique(outer_points_arr)

        return outer_points_arr

    # Returns an array where each element is a 2-tuple of the for (pixel_quintuple, pixel_quintuple)
    # This tuple represents the pixels on either edge of the rim, for both rows and columns
    def get_inner_outer_pixel_pairs_list(self):
        start_row, end_row = self.longest_4_intersections_rows_series_range
        start_col, end_col = self.longest_4_intersections_columns_series_range

        pixel_tuple_list = []

        for i in range(start_row, end_row+1):
            current_strip = self.augmented_rows_list[i]
            ordered_white_pixels = current_strip.get_ordered_white_pixels_list()

            assert (len(ordered_white_pixels) == 4)
            tuple_1 = ordered_white_pixels[1], ordered_white_pixels[0]
            tuple_2 = ordered_white_pixels[2], ordered_white_pixels[3]
            pixel_tuple_list.append(tuple_1)
            pixel_tuple_list.append(tuple_2)

        for i in range(start_col, end_col+1):
            current_strip = self.augmented_columns_list[i]
            ordered_white_pixels = current_strip.get_ordered_white_pixels_list()

            assert (len(ordered_white_pixels) == 4)
            tuple_1 = ordered_white_pixels[1], ordered_white_pixels[0]
            tuple_2 = ordered_white_pixels[2], ordered_white_pixels[3]
            pixel_tuple_list.append(tuple_1)
            pixel_tuple_list.append(tuple_2)

        return pixel_tuple_list


class Line:
    def __init__(self, p1, p2):
        assert p1 is not p2
        assert p1[0] != p2[0]

        slope = (p1[1] - p2[1])/(p1[0] - p2[0])
        intercept = p1[1] - slope*p1[0]

        self.slope = slope
        self.intercept = intercept

    def compute_y(self, x):
        return self.slope*x + self.intercept

    def is_below_point(self, p):
        x, y = p[0], p[1]
        line_is_below = self.compute_y(x) < y

        return line_is_below

    def distance_from_point(self, p):
        x_0 = p[0]
        y_0 = p[1]

        x_intercept = (self.slope*y_0 + x_0 - self.slope*self.intercept) / (1 + self.slope**2)
        y_intercept = x_intercept*self.slope + self.intercept

        distance = math.sqrt((x_intercept - x_0)**2 + (y_intercept - y_0)**2)
        return distance


class RimFinder:
    def __init__(self, binary_img):
        self.augmented_image = AugmentedBinaryImage(binary_img)

    def get_inner_ellipse(self):
        contours = self.__convert_to_contours(self.augmented_image.get_inner_rim_points_quintuples_arr())

        ellipse = cv2.fitEllipse(contours)

        return ellipse

    # Wraps the pixel quintuples into the opencv friendly "contour" format
    @staticmethod
    def __convert_to_contours(point_quintuples):
        contour_list = []

        for pixel in point_quintuples:
            first = np.array([pixel['column'], pixel['line']], dtype=np.int32)
            second = np.empty([1, 2], dtype=np.int32)
            second[0] = first
            contour_list.append(second)

        return np.array(contour_list)

    def get_major_axis_line(self):
        inner_ellipse = self.get_inner_ellipse()

        # OpenCV has upside down y-coordinates, so we just flip it for us normal people
        original_center = inner_ellipse[0]

        cartesian_x = original_center[0]
        cartesian_y = self.augmented_image.height - original_center[1]

        # Now use some trigonometry to find a second point on the long axis
        axis_lengths = inner_ellipse[1]
        delta_y = min(axis_lengths[0], axis_lengths[1]) / 2

        angle_degrees = inner_ellipse[2]
        angle_radians = angle_degrees/360 * 2*math.pi

        delta_x = delta_y * math.tan(angle_radians)

        p2 = cartesian_x - delta_x, cartesian_y - delta_y
        p1 = cartesian_x, cartesian_y

        major_axis_line = Line(p1, p2)
        return major_axis_line

    # TODO find a more scientific cutoff distance
    def find_rim(self, ratio_error_acceptance=0.02):
        points_pairs_list = self.augmented_image.get_inner_outer_pixel_pairs_list()
        long_axis_line = self.get_major_axis_line()
        inner_ellipse = self.get_inner_ellipse()
        cutoff_distance = min(inner_ellipse[1]) * 0.15

        # if the inner ellipse is a circle, we can just use that
        inner_ellipse_major_axis_length = inner_ellipse[1][0]
        inner_ellipse_minor_axis_length = inner_ellipse[1][1]
        ratio = inner_ellipse_major_axis_length / inner_ellipse_minor_axis_length
        if 1 + ratio_error_acceptance > ratio > 1 - ratio_error_acceptance:
            return inner_ellipse
        
        rim_points_list_jagged = []

        for pair in points_pairs_list:
            inner_pixel = pair[0]
            outer_pixel = pair[1]

            if long_axis_line.is_below_point((inner_pixel['x'], inner_pixel['y'])):
                rim_points_list_jagged.append(outer_pixel)
            else:
                rim_points_list_jagged.append(inner_pixel)

        rim_points_list_smooth = []

        for point in rim_points_list_jagged:
            cartesian_coords = point['x'], point['y']
            if long_axis_line.distance_from_point(cartesian_coords) > cutoff_distance:
                rim_points_list_smooth.append(point)

        true_ellipse_contours = self.__convert_to_contours(rim_points_list_smooth)

        true_ellipse = cv2.fitEllipse(true_ellipse_contours)
        return true_ellipse




    
















