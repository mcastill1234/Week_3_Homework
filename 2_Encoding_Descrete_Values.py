import numpy as np
import Perceptron_From_HW2 as ph2
import HyperplaneProcedures as hp

data = np.array([[2, 3, 4, 5]])
labels = np.array([[1, 1, -1, -1]])

# Problem 2A
th, th0 = ph2.perceptron(data, labels)
print("Answer 2A: (thetha, thetha_0) = ", th, th0)

# Problem 2B
new_data = np.array([[1, 6]])
prediction1 = ph2.positive(new_data, th, th0)
print("Answer 2B: The prediction on Samsung and Nokia phones is: ", prediction1)


# Problem 2D
def one_hot(x, k):
    encoded = np.zeros((k, 1))
    encoded[x-1, 0] = 1
    return encoded


def one_hot_encoder(data1, k):
    dim, num_sam = data1.shape
    encoded_data = np.zeros((k, num_sam))
    for i in range(num_sam):
        encoded_data[:, i:i+1] = one_hot(data1[0, i], 6)
    return encoded_data


# Problem 2E i
encoded_dataset = one_hot_encoder(data, 6)
th1, th0_1 = ph2.perceptron(encoded_dataset, labels)
print("Answer 2E i: theta = ", th1.T, "theta_0 = ", th0_1)

# Problem 2E ii
new_data1 = np.array([[1, 6]])
new_encoded = one_hot_encoder(new_data1, 6)
prediction2 = ph2.positive(new_encoded, th1, th0_1)
print("Answer 2E ii: Prediction on Samsung and Nokia data points: ", prediction2)

# Problem 2E iii
dist1 = hp.signed_dist(new_encoded, th1, th0_1)
print("Answer 2E iii: The distances for 2E ii predictions from the separator are: ", dist1)


# Problem 2G
data_2G = np.array([[1, 2, 3, 4, 5, 6]])
labels_2G = np.array([[1, 1, -1, -1, 1, 1]])
encoded_data_2G = one_hot_encoder(data_2G, 6)
th2, th0_2 = ph2.perceptron(encoded_data_2G, labels_2G)
print("Answer 2G: One Hot encoded data produces: Theta =", th2.T, "Theta_0 = ",  th0_2)