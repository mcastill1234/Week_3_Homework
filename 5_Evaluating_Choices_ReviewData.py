import numpy
import numpy as np

import code_for_hw3_part2 as hw3


# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = hw3.load_review_data('reviews.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))

# The dictionary of all the words for "bag of words"
dictionary = hw3.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = hw3.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = hw3.rv(review_label_list)
# print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)

# print()
# print("Answers to 5.1A to 5.2B")
# for i in [1, 10, 50]:
#     learner_score1 = hw3.xval_learning_alg(hw3.perceptron, review_bow_data, review_labels, 10, params={'T': i})
#     print("Perceptron score for T = ", i, "Feature set 1: ", learner_score1)
#     learner_score2 = hw3.xval_learning_alg(hw3.averaged_perceptron, review_bow_data, review_labels, 10, params={'T': i})
#     print("Averaged Perceptron score for T = ", i, "Feature set 1: ", learner_score2)
#     print()

# Problems 5.2A 5.2B
th, th0 = hw3.averaged_perceptron(review_bow_data, review_labels, params={'T': 10})
# reverse = hw3.reverse_dict(dictionary)
# sorted_indices = (-th.T).argsort()
# most_positive = []
# most_negative = []
# for i in range(10):
#     most_positive.append(reverse.get(sorted_indices[0, i]))
#     most_negative.append(reverse.get(sorted_indices[0, -i-1]))
# print("Answer 5.2A: Most positive words: ", most_positive)
# print("Answer 5.2B: Most negative words: ", most_negative)


# Problem 5.2C
d, n = review_bow_data.shape
bow_data_lengths = hw3.signed_dist(review_bow_data, th, th0)
print("Answer 5.2C:")
print("The most positive review is: ")
print(review_texts[np.argmax(bow_data_lengths)])
print()
print("The most negative review is: ")
print(review_texts[np.argmin(bow_data_lengths)])







