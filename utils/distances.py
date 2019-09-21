import numpy as np


def vector_length(v):
  return np.sqrt(np.dot(v, v))

def euclidean_distance(u, v):
  return vector_length(u - v)

def normalized_length(v):
  return v / vector_length(v)

def cosine_distance(u, v):
  return 1.0 - (np.dot(u, v) / (vector_length(u) * vector_length(v)))

def intersection(u, v):
  return np.sum(np.minimum(u, v))

def union(u, v):
  return np.sum(np.maximum(u, v))

def jaccard_distance(u, v):
  return 1.0 - (intersection(u, v) / union(u, v))

def jensen_shannon_distance(u, v):
  return np.sum(u * np.log(u / v)) ** 2.0

def dice_coefficient(u, v):
  return 1.0 - (2.0 * np.sum(np.minimum(u, v)) / np.sum(u + v))

def soft_cosine_similarity(u, v):
  return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))

def distances(word, matrix, word2row, row2word, metric=euclidean_distance):
  word_distances = {}
  for i, row in enumerate(matrix):
    distance = metric(matrix[word2row[word]], row)
    word_distances[row2word[i]] = distance
    # print(f"{' '.join(str(metric.__name__).capitalize().split('_'))} between {word} & {row2word[i]}: {distance}")
  return word_distances

def distances_matrix(matrix, metric=euclidean_distance):
  return [[metric(rowi, rowj) for rowj in matrix] for rowi in matrix]

def top_most_similar_words(word, matrix, top=10, _distances=False):
  word_distances_row = matrix[word2row[word]]
  sorted_top_words = sorted([[value, row2word[i]] for value, i in zip(word_distances_row, range(len(word_distances_row)))], key=lambda l: l[0])
  if _distances == True:
    return [pair[0] for pair in sorted_top_words][1:top+1], [pair[1] for pair in sorted_top_words][1:top+1]
  return [pair[1] for pair in sorted_top_words][1:top+1]

def find_word_with_the_closest_words_from_all(matrix):
  min_mean = 9999999999.0
  min_word = ""
  min_list_of_words = []
  for row in range(len(matrix)):
    top_values, top_words = top_most_similar_words(row2word[row], matrix, 6, True)
    curr_mean = np.mean(top_values)
    if curr_mean < min_mean:
      min_mean = curr_mean
      min_word = row2word[row]
      min_list_of_words = top_words
  return min_word, min_list_of_words

