import numpy as np

# Takes as input a 3xsomething matrix of POSITIVE floats
# First row is x coordinate, second row is y coordinate, third row is ball size

test_matrix = np.array([[1, 2, 3, 4, 5],
                           [2, 4, 6, 8, 10],
                           [1, 1, 1, 1, 1]], dtype=float)


def get_regression_line(shot_matrix):
    slope, intercept = np.polyfit(shot_matrix[0], shot_matrix[1], 1)
    return slope, intercept


# Given a point and a line, projects the point onto the line and returns that point of the line
def orthogonal_projection_point(slope, intercept, x, y):
    x_line = (slope*y + x - intercept*slope)/(slope**2+1)
    y_line = slope*x_line + intercept

    return x_line, y_line


# Given a shot matrix, fits a line, and reassigns all the x,y positions to points on the line, and assigns them the appropriate distance
def align_matrix(raw_shot_matrix):
    assert raw_shot_matrix.dtype == float

    slope, intercept = get_regression_line(raw_shot_matrix)

    aligned_shot_matrix = np.copy(raw_shot_matrix)

    for i in range(len(raw_shot_matrix[0])):
        x_line, y_line = orthogonal_projection_point(slope, intercept, raw_shot_matrix[0,i], raw_shot_matrix[1,i])
        aligned_shot_matrix[0, i] = x_line
        aligned_shot_matrix[1, i] = y_line

    return aligned_shot_matrix


# Put a negative sign in front since when ball is close to camera, we want it on top of the parabola
def get_z_coordinate(object_diameter, image_diameter, focal_length, camera_height):
    z = camera_height - object_diameter * focal_length / image_diameter

    return z


# Returns a matrix with three rows: x, y, approximate z
def format_matrix(raw_shot_matrix, object_diameter, focal_length, camera_height):
    coordinate_matrix = align_matrix(raw_shot_matrix)

    for i in range(len(coordinate_matrix[2])):
        coordinate_matrix[2, i] = get_z_coordinate(object_diameter, coordinate_matrix[2,i], focal_length, camera_height)

    return coordinate_matrix









