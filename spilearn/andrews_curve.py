# coding: utf-8

import numpy as np


class AndrewsCurve:
    def __init__(self, vector, rad, h):
        self._rad = rad
        self._h = h
        self._accuracy = rad / h
        self._vector = vector

    def theta(self):
        return np.linspace(0, self._rad, self._accuracy)

    def curve(self):
        curve = list()
        for th in self.theta():
            j = 1
            f = self._vector[0] / np.sqrt(2)
            for i in range(1, len(self._vector), 2):
                f += self._vector[i] * np.sin(j * th)
                if i + 1 <= len(self._vector[1::2]):
                    f += self._vector[i + 1] * np.cos(j * th)
                j += 1
            curve.append(f)
        return curve

    def discrete_curve(self):
        return (
            np.around(self.curve()).astype(int) + 1
        )  # + np.abs(np.amin(np.around(self.curve())))


if __name__ == '__main__':
    pass
