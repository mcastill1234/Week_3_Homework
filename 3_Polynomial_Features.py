import numpy as np
from code_for_hw3_part1 import make_polynomial_feature_fun


# Problem 3A
orders = [1, 10, 20, 30, 40, 50]
answer = []
for order in orders:
    feature_transform = make_polynomial_feature_fun(order)
    answer.append(feature_transform(np.array([[1, 1]]).T).shape[0])
print("Answer 3A: The number of polynomial features of degrees", orders, "is: ", answer)

# Problem 3B
print("Answer 3B: After running test_with_features for the given data sets we found the following:")
print("super_simple_separable_through_origin: min Order = 1")
print("super_simple_separable: min order =  1")
print("xor: min order = 2")
print("xor_more: min order = 3")