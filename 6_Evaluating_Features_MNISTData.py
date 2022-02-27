import code_for_hw3_part2 as hw3
from code_for_hw3_part2 import cv
import numpy as np

mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)

# HINT: change the [0] and [1] if you want to access different images
d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1, -1)
y1 = np.repeat(1, len(d1)).reshape(1, -1)

# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T


# Problem 6.1A
def row_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (m,1) array where each entry is the average of a row
    """
    return np.mean(x, axis=1, keepdims=True)


# Simple test for problem 6.1A
a = np.arange(9).reshape(3, 3)
print("Test data for problem 6.1A:\n", a)
b = row_average_features(a)
print("Row average features for problem 6.1A:\n", b)


# Problem 6.1B
def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    return np.mean(x, axis=0, keepdims=True).T


# Simple test for problem 6.1B
print("Test data for problem 6.1B:\n", a)
d = col_average_features(a)
print("Row average features for problem 6.1B:\n", d)


# Problem 6.1C
def top_bottom_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (2,1) array where the first entry is the average of the
    top half of the image = rows 0 to floor(m/2) [exclusive]
    and the second entry is the average of the bottom half of the image
    = rows floor(m/2) [inclusive] to m
    """
    m = x.shape[0]
    return cv([np.mean(x[:m // 2, ]), np.mean(x[m // 2:, ])])


# Simple test for problem 6.CB
print("Test data for problem 6.1C:\n", a)
e = top_bottom_features(a)
print("Row average features for problem 6.1C:\n", e)