# Module is create to take a 3 row matrix, and return a parametric curve for the shot
# Row 0 should be x-positions, row 2 should be y-positions, row 3 should be approximate z-positions
# The z-positions are expected to not be perfectly parabolic, but the x and y coords should be on some y=mx + b line

import numpy as np
import math

import matplotlib.pyplot as plt

test_matrix = np.array([[1, 2, 3, 4, 5],
                           [2, 4, 6, 8, 10],
                           [1, 1, 1, 1, 1]], dtype=float)

# Reduces 3-D shot to equivalent 2-D coords
def flatten_shot(shot_matrix):
    squished = np.array([shot_matrix[0], shot_matrix[2]])

    for i in range(len(shot_matrix[0])):

        squished[0, i] = math.sqrt(shot_matrix[0, i]**2 + shot_matrix[1, i]**2)

    return squished


# Performs the polynomial fitting
def fit_parabola(r_arr, z_arr):
    return np.polyfit(r_arr, z_arr, 2)


# Given the 3 coefficients of a 2D parabola, proceeds to parameterize it
# Takes coefficients in decreasing order of degrees
def parameterize_2d_parabola(a, b, c):
    r = np.polynomial.polynomial.Polynomial([0, 1])
    z = np.polynomial.polynomial.Polynomial([c, b, a])

    return np.array([r, z])


def parameterize_2d_line(x_arr, y_arr):
    directing_vector = [(x_arr[1] - x_arr[0]), (y_arr[1] - y_arr[0])]

    x = np.polynomial.polynomial.Polynomial([x_arr[0], directing_vector[0]])
    y = np.polynomial.polynomial.Polynomial([y_arr[0], directing_vector[1]])

    return np.array([x, y])


def distance_between_points(p1, p2):
    x_diff = p1[0] - p2[0]
    y_diff = p1[1] - p2[1]

    dist = math.sqrt(x_diff**2 + y_diff**2)
    return dist


# This implementation is absolutely fucked, since the x polynomial and y polynomial have a different variable
# than the z polynomial. This class is meant to be a black box so that the user doesn't need to worry about fixing it
# Note that t = sqrt(v^2 +(mv + b)^2)
# raw matrix contains x-positions, y-positions, ball size rows
# There should be a pdf/onenote somewhere explaining the math

class ParametricParabola:
    def __init__(self, aligned_matrix):

        flat_shot = flatten_shot(aligned_matrix)

        parabola_coefficients = fit_parabola(flat_shot[0], flat_shot[1])
        parabola_coefficients = list(map(lambda x : round(x, 5), parabola_coefficients))

        parametric_parabola_2d = parameterize_2d_parabola(parabola_coefficients[0], parabola_coefficients[1], parabola_coefficients[2])

        parametric_line_xy = parameterize_2d_line(aligned_matrix[0], aligned_matrix[1])

        self.x_poly_base_t = parametric_line_xy[0]
        self.y_poly_base_t = parametric_line_xy[1]
        self.z_poly_base_r = parametric_parabola_2d[1]
        self.__zeroes_distance = None
        self.__xy_zeroes = None
        self.__parameter_range = None

    def evaluate_at(self, t):
        x = self.x_poly_base_t(t)
        y = self.y_poly_base_t(t)

        r = math.sqrt(x**2 + y**2)
        z = self.z_poly_base_r(r)

        return np.array([round(x, 3), round(y, 3), round(z, 3)])

    def __get_all_potential_xy_zeroes(self):
        zeroes = self.z_poly_base_r.roots()
        self.__zeroes_distance = max(zeroes) - min(zeroes)

        potential_xy_zeroes = []

        for zero in zeroes:
            temp_poly = self.x_poly_base_t**2 + self.y_poly_base_t**2 - zero**2
            t_s = temp_poly.roots()

            for t in t_s:
                x = self.x_poly_base_t(t)
                y = self.y_poly_base_t(t)
                potential_xy_zeroes.append((x, y))

        return potential_xy_zeroes

    # Can throw an exception
    def __narrow_down_by_direction(self):
        potential_zeroes = self.__get_all_potential_xy_zeroes()

        assert len(potential_zeroes) <= 4

        if len(potential_zeroes) == 2:
            return potential_zeroes

        if len(potential_zeroes) < 2:
            raise Exception("The polynomial had less than 2 roots, i.e. it doesn't follow projectile motion")

        # We now have the zeroes ordered from furthest away to closest away
        # List can be 3 or four elements long, depending on the parabola
        potential_zeroes.sort(key=lambda p: p[0]**2 + p[1]**2, reverse=True)

        candidate_vector_1 = (potential_zeroes[0][0] - potential_zeroes[-1][0], potential_zeroes[0][1] - potential_zeroes[-1][1])

        xy_line_directing_vector = (self.x_poly_base_t.coef[1], self.y_poly_base_t.coef[1])

        # We need to check if these 2 vectors ar in the same direction. We check if dot product is positive
        dot_product = candidate_vector_1[0]*xy_line_directing_vector[0] + candidate_vector_1[1]*xy_line_directing_vector[1]

        if dot_product > 0:
            potential_zeroes.pop(1)
        else:
            potential_zeroes.pop(0)

        return potential_zeroes

    # Can throw an exception
    def __narrow_down_by_dist(self):
        # Ordered from furthest to nearest from origin
        ordered_narrowed_down_zeroes = self.__narrow_down_by_direction()

        if len(ordered_narrowed_down_zeroes) == 2:
            return ordered_narrowed_down_zeroes

        # If we make it here, there should be three potential zeroes
        dist1 = distance_between_points(ordered_narrowed_down_zeroes[0], ordered_narrowed_down_zeroes[1])
        dist2 = distance_between_points(ordered_narrowed_down_zeroes[0], ordered_narrowed_down_zeroes[2])

        if abs(dist1 - self.__zeroes_distance) < abs(dist2 - self.__zeroes_distance):
            ordered_narrowed_down_zeroes.pop(2)
        else:
            ordered_narrowed_down_zeroes.pop(1)

        return ordered_narrowed_down_zeroes

    # Can throw an exception
    # TODO: currently propagates the exception forward. Convenient for testing, but needs to be fixed eventually
    def get_xy_zeroes(self):
        if self.__xy_zeroes is None:
            self.__xy_zeroes = self.__narrow_down_by_dist()

        return self.__xy_zeroes

    # Currently throws an exception, which is not suitable. Will fix itself when get_xy_zeroes handles exception
    # TODO: handle the case where there are no zeroes (i.e. non-parabolic path)
    def get_parameter_range(self):
        if self.__parameter_range is None:
            xy_zeroes = self.get_xy_zeroes()

            parameter_bound_1 = (self.x_poly_base_t - xy_zeroes[0][0]).roots()[0]
            parameter_bound_2 = (self.x_poly_base_t - xy_zeroes[1][0]).roots()[0]

            self.__parameter_range = sorted([parameter_bound_1, parameter_bound_2])

        return self.__parameter_range

    def get_3d_points_arr(self, amount=100):
        parameter_range = self.get_parameter_range()
        t_values = np.linspace(parameter_range[0], parameter_range[1], num=amount)

        shot_points_arr = np.empty([3, amount])

        for i in range(amount):
            shot_points_arr[:,i] = self.evaluate_at(t_values[i])

        return shot_points_arr


r = math.sqrt(2)

test3 = np.array([[r/2, r, 3*r/2, 2*r, 5*r/2],
                 [r/2, r, 3*r/2, 2*r, 5*r/2],
                 [0, 3, 4, 3, 0]])

test_parabola = ParametricParabola(test3)
position_matrix = test_parabola.get_3d_points_arr()


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(position_matrix[0], position_matrix[1], position_matrix[2], 'red')
plt.show()






