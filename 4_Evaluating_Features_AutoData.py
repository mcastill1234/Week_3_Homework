import code_for_hw3_part2 as hw3
import numpy as np

# Problems 4.1A to 4.1E
# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = hw3.load_auto_data('auto-mpg.tsv')

features1 = [('cylinders', hw3.raw),
            ('displacement', hw3.raw),
            ('horsepower', hw3.raw),
            ('weight', hw3.raw),
            ('acceleration', hw3.raw),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.raw)]
auto_data1, auto_labels1 = hw3.auto_data_and_labels(auto_data_all, features1)

features2 = [('cylinders', hw3.one_hot),
            ('displacement', hw3.standard),
            ('horsepower', hw3.standard),
            ('weight', hw3.standard),
            ('acceleration', hw3.standard),
            ## Drop model_year by default
            ## ('model_year', hw3.raw),
            ('origin', hw3.one_hot)]

auto_data2, auto_labels2 = hw3.auto_data_and_labels(auto_data_all, features2)

print()
print("Answers to 4.1A to 4.1E")
for i in [1, 10, 50]:
    learner_score1 = hw3.xval_learning_alg(hw3.perceptron, auto_data1, auto_labels1, 10, params={'T': i})
    print("Perceptron score for T = ", i, "Feature set 1: ", learner_score1)
    learner_score2 = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data1, auto_labels1, 10, params={'T': i})
    print("Averaged Perceptron score for T = ", i, "Feature set 1: ", learner_score2)
    print()
    learner_score3 = hw3.xval_learning_alg(hw3.perceptron, auto_data2, auto_labels2, 10, params={'T': i})
    print("Perceptron score for T = ", i, "Feature set 2 : ", learner_score3)
    learner_score4 = hw3.xval_learning_alg(hw3.averaged_perceptron, auto_data2, auto_labels2, 10, params={'T': i})
    print("Averaged Perceptron score for T = ", i, "Feature set 2: ", learner_score4)
    print()

# Problem 4.2A
th, th0 = hw3.averaged_perceptron(auto_data2, auto_labels2, params={'T': 50})
print(th.T, th0)

