# This k-NN algorithm is based on the work implemented by Jason Brownlee
# The code below has been adapted in order to soften the decision boundary

# Original work taken from: Develop k-Nearest Neighbors in Python From Scratch,
#                           Machine Learning Mastery
# URL: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

from csv import reader
from math import sqrt


# from numpy array to list
def np_to_list(l):
    new_list = []
    for attr in l:
        new_list.append(float(attr))
    return new_list


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    temp_row1 = np_to_list(row1)
    temp_row2 = np_to_list(row2)
    # print("temp ROW1: ", temp_row1)
    # print("temp ROW2: ", temp_row2)
    distance = 0.0
    for i in range(len(temp_row1) - 1):
        distance += (temp_row1[i] - temp_row2[i]) ** 2
    return sqrt(distance)


class KNN:

    # Locate the most similar neighbors
    def get_neighbors(self, train, test_row, num_neighbors):
        distances = list()
        for train_row in train:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    # Make a prediction with neighbors
    def predict_classification(self, train, test_row, num_neighbors):
        neighbors = self.get_neighbors(train, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return neighbors
