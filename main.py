import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice


# image = io.imread('example.png')
#
# print(f"image = {image.shape}")
# print("image.dtype = {}".format(image.dtype))
#
# exit()
from tqdm import tqdm


def numeric_value_to_image_value(value):
    """
    :param value: numeric in [0, 1]
    :return: uint8 in [0, 255]
    """
    result = np.uint8(round(value * 255))
    assert 0 <= result <= 255
    return result


def get_random_neighbor_point(x, y):
    dx, dy = random.choice([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    return dx + x, dy + y


def get_random_neighbor_not_going_backward(initial_point, center_point, max_distance_to_center_can_loose=0):
    initial_distance_to_center = l1_distance(initial_point, center_point)

    while True:
        new_neighbor = get_random_neighbor_point(*initial_point)
        new_neighbor_distance_to_center = l1_distance(new_neighbor, center_point)
        if initial_distance_to_center - max_distance_to_center_can_loose <= new_neighbor_distance_to_center:
            return new_neighbor


def l1_distance(point1, point2):
    """
    :param point1: (x, y)
    :param point2: (x, y)
    :return: l1 distance
    """
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


class Fractal:
    def __init__(self, steps_count=0, width=200, height=150):
        self.steps_count = steps_count

        self.width = width
        self.height = height

        self.dict_xy_points = {(0, 0): 1.}  # dict (x,y) => luminosity
        self.last_step_created_points = [(0, 0)]

    def run(self):
        for _ in tqdm(range(self.steps_count)):
            self.step()

    def step(self):
        new_points = []
        for last_p in self.last_step_created_points:
            new_points_count = Fractal.get_points_to_create_count()

            # create new next_point
            for _ in range(new_points_count):
                proba1, proba2 = 1., 2.
                sum_proba = proba1 + proba2
                max_distance_to_center_can_loose = choice([0, 1], 1, p=[proba1/sum_proba, proba2/sum_proba])[0]
                new_point = get_random_neighbor_not_going_backward(last_p,
                                                                   (0, 0),
                                                                   max_distance_to_center_can_loose=max_distance_to_center_can_loose)
                new_points.append(new_point)
                self.dict_xy_points[new_point] = 1.
        self.last_step_created_points = new_points

    @staticmethod
    def get_points_to_create_count():
        proba1, proba2 = 200., 1.
        sum_proba = proba1 + proba2
        return choice([1, 2], 1, p=[proba1/sum_proba, proba2/sum_proba])[0]

    def tmp_random(self):
        for x in range(0, self.width):
            for y in range(0, self.height):
                self.dict_xy_points[(x, y)] = random.uniform(0, 1)

    def to_image(self, save=True):
        result = np.zeros((self.width, self.height, 3), dtype="uint8")
        for (x, y), points in self.dict_xy_points.items():
            image_x = x + round(self.width / 2)
            image_y = y + round(self.height / 2)

            # ignore points not in image
            if not 0 <= image_x < self.width:
                continue
            if not 0 <= image_y < self.height:
                continue

            result[image_x][image_y][1] = numeric_value_to_image_value(points)

        if save:
            plt.imsave('result.png', result, cmap=cm.gray)
            plt.imshow(result)
            plt.show()


def main():
    fr = Fractal(width=1200, height=800, steps_count=1200)
    fr.run()
    print("creating image")
    # fr.tmp_random()
    fr.to_image()


if __name__ == "__main__":
    main()
