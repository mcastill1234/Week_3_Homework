import numpy as np
import HyperplaneProcedures as hp


def perceptron_through_origin(data, labels):
    dim, num = data.shape
    theta = np.zeros((dim, 1))
    mistakes = 0
    for tao in range(500000):
        changed = False
        for i in range(num):
            y_i = labels[0, i]
            x_i = data[:, i]
            current_guess = y_i * np.dot(theta.T, x_i)
            if current_guess <= 0:
                mistakes += 1
                theta = theta + hp.cv(y_i * x_i)
                changed = True
                # print("New theta = ", theta.T)
        if not changed:
            break
    # print("Number of mistakes: ", mistakes)
    # print()
    return theta, mistakes


# Problem 1A:
dataset1 = np.array([[200, 800, 200, 800], [0.2, 0.2, 0.8, 0.8], [1, 1, 1, 1]])
np.set_printoptions(suppress=True)
# print("dataset1: \n", dataset1)
labelset1 = np.array([[-1, -1, 1, 1]])
theta1 = np.array([[0, 1, -0.5]])
dist2hp1 = labelset1 * (theta1 @ dataset1) / hp.length(theta1)
# print(theta1 @ dataset1)
margin1 = np.min(dist2hp1)
print("Answer 1a: The margin of this data set is: ", margin1)

# Problem 1B:
radius1 = (800 ** 2 + 0.8 ** 2 + 1 ** 2) ** 0.5
mistake_bound1 = (radius1 / margin1) ** 2
print("Answer 1b: The mistake bound is: ", mistake_bound1)

# Problem 1C:
# th1, mis1 = perceptron_through_origin(dataset1, labelset1)
# print("Answer 1c: The number of mistakes is: ", mis1)

# Problem 1D:
dataset2 = np.array([[0.001, 0.001, 1]]).T * dataset1
# print("dataset2: \n", dataset2)
theta2 = np.array([[0, 1, -0.0005]])
dist2hp2 = labelset1 * (theta2 @ dataset2) / hp.length(theta2)
# print(theta2 @ dataset2)
margin2 = np.min(dist2hp2)
print("Answer 1d: The margin when scaling the data set by 0.001 is: ", margin2)

# Problem 1E:
radius2 = (0.8 ** 2 + 0.008 ** 2 + 1 ** 2) ** 0.5
mistake_bound2 = (radius2 / margin2) ** 2
print("Answer 1e: The new mistake bound is: ", mistake_bound2)

# Problem 1F:
dataset3 = np.array([[0.001, 1, 1]]).T * dataset1
dist2hp3 = labelset1 * (theta1 @ dataset3) / hp.length(theta1)
margin3 = np.min(dist2hp3)
print("Answer 1f: The margin when scaling only the first feature by 0.001 is: ", margin3)

# Problem 1G:
radius3 = (0.8 ** 2 + 0.8 ** 2 + 1 ** 2) ** 0.5
mistake_bound3 = (radius3 / margin3) ** 2
print("Answer 1e: The new mistake bound is: ", mistake_bound3)

# Problem 1H:
th3, mis3 = perceptron_through_origin(dataset3, labelset1)
print("Answer 1H: The number of mistakes using scaled first feature is: ", mis3)