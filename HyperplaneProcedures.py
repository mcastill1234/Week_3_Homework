# Introduction to numpy from 6.036 - Useful functions from week 1 exercises and homework 1

import numpy as np


def rv(value_list):
    """Takes a list of numbers and returns a row vector: 1 x n"""
    return np.array([value_list])


def cv(value_list):
    """Takes a list of numbers and returns a column vector: n x 1"""
    return rv(value_list).T


def length(col_v):
    """Takes a d by 1 matrix, returns a scalar of their lengths"""
    return np.sum(col_v * col_v) ** 0.5


def normalize(col_v):
    """Takes a column vector n x 1 and returns a unit vector in the same direction"""
    return col_v / length(col_v)


def signed_dist(x, th, th0):
    """Takes column vector 'x' of (d x 1), th of the same dimension and a scalar th0 and returns the signed
    perpendicular distance as a 1 x 1 array from the hyperplane encoded by (th, th0) to x."""
    return (th.T@x + th0) / length(th)


def positive(x, th, th0):
    """Takes column vectors 'x' and th of (d x 1), a scalar th0 and returns +1 if x is on the positive side of the
    hyperplane encoded by (th, th0), 0 if on the hyperplane, -1 otherwise."""
    return np.sign(np.dot(th.T, x) + th0)


def score(data, labels, th, th0):
    """Takes data (d x n) representing n data points in d dimensions, labels (1 x n) array of
    elements in (+1, -1), representing target labels, th (d x 1) and scalar th0 represent a hyperplane.
    Returns the number of points for which the label is equal to the output of the positive function on the point"""
    return np.sum(positive(data, th, th0) == labels)


def best_separator(data, labels, ths, th0s):
    """Takes data (d x n) representing n data points in d dimensions, labels (1 x n) array of elements in (+1, -1),
    representing target labels, ths (d x m) and th0s (1 x m) represent the candidate hyperplanes.
    Returns the hyperplane with the highest score in a tuple (d x 1) array and an offset in the form of 1 by 1 array."""
    best_index = np.argmax(score_mat(data, labels, ths, th0s))
    return cv(ths[:, best_index]), th0s[:, best_index:best_index+1]


def score_mat(data, labels, ths, th0s):
    pos = np.sign(ths.T@data + th0s.T)
    return np.sum(pos == labels, axis=1, keepdims=True)




