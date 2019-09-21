import os
import numpy as np
from pprint import pprint

import utils.preprocessing as preprocessing
import utils.distances as metrics
from devmood.nn.learning.nn_learner import learn


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TESTING DISTANCE METRICS:               |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# triangle = np.array([[ 2.0,  4.0], [10.0, 15.0], [14.0, 10.0]])
# for distance_metric in (metrics.euclidean_distance, metrics.cosine_distance, metrics.jaccard_distance, metrics.jensen_shannon_distance, metrics.dice_coefficient, metrics.soft_cosine_similarity):
#   print(f"{distance_metric.__name__}")
#   print(f"|AB| = {distance_metric(triangle[0], triangle[1])}")
#   print(f"|BC| = {distance_metric(triangle[1], triangle[2])}")
#   print(f"|AC| = {distance_metric(triangle[0], triangle[2])}", end="\n")
# exit(12)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GETTING COOCURRENCE MATRIX:             |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# text_splitted_to_docs = preprocessing.preprocess_pan_tadeusz()
# docs2sentences2tokens = preprocessing.tokenize_docs_sentences(text_splitted_to_docs)
# word2idx, tcp_idx, vocab_idx, pair2count = preprocessing.target_context_pairs_operations(docs2sentences2tokens)
# idx2word = preprocessing.invert_dict(word2idx)
# coocurrence_dict = preprocessing.get_word_coocurrence_dict(word2idx, tcp_idx, vocab_idx, pair2count)
# coocurrence_matrix_np = preprocessing.get_coocurrence_matrix(vocab_idx, coocurrence_dict)
# word2row = {idx2word[idx]: row for idx, row in zip(vocab_idx, range(len(vocab_idx)))}
# row2word = preprocessing.invert_dict(word2row)
# 
# ksiadz_row = list(coocurrence_matrix_np[word2row["ksiądz"]])
# print(set(ksiadz_row))
# # KSIĄDZ WITH ROBAK - 9 COOCURRANCES:
# print([row2word[row] for row, value in zip(range(len(ksiadz_row)), ksiadz_row) if value == 9])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PREVIEWING COOCURRENCE MATRIX:          |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# preprocessing.preview_coocurance_matrix(coocurrence_matrix_np, row2word)
# preprocessing.preview_coocurance_matrix(coocurrence_matrix_np, row2word, "ksiądz")
# exit(12)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PyTorch SkipGram MODEL TRAINING:        |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# args = {"batch_size": 128, "dims": 100, "eta": 0.001, "window": 5, "epochs": 50, "input": "Pan Tadeusz"}
# learn(args)
# exit(12)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOADING NEWEST PyTorch WORD EMBEDDINGS: |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# weights_path = os.path.join(os.getcwd(), "devmood/nn/results/weights")
# word_embeddings = [[pair[0], [float(vecval) for vecval in pair[1].split(" ")]] for row in [[row.split(",") for row in open(os.path.join(weights_path, sorted(os.listdir(weights_path))[-1]), encoding="utf-8").read().split("\n") if row != ""]] for pair in row]
# print(word_embeddings[0])
# exit(12)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LOADING GENSIM PRETRAINED EMBEDDINGS:   |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

word_embeddings = [[pair[0], [float(vecval) for vecval in pair[1].split(" ")]] for row in [[row.split(",") for row in open(os.path.join(os.getcwd(), "devmood/nn/results/word-embedds-gensim-500d-500e-cbow.txt"), encoding="utf-8").read().split("\n") if row != ""]] for pair in row]
# word_embeddings = [[pair[0], [float(vecval) for vecval in pair[1].split(" ")]] for row in [[row.split(",") for row in open(os.path.join(os.getcwd(), "devmood/nn/results/word-embedds-gensim-100d-50e-sg.txt"), encoding="utf-8").read().split("\n") if row != ""]] for pair in row]
# print(word_embeddings[0])
# exit(12)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GETTING DISTANCES MATRIX:               |
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

word2row = {word: idx for idx, word in enumerate([x[0] for x in word_embeddings])}
row2word = {idx: word for word, idx in word2row.items()}

word_embeddings = [np.array(row[1]) for row in word_embeddings]
ksiadz_distances = metrics.distances("ksiądz", word_embeddings, word2row, row2word, metrics.soft_cosine_similarity)
# exit(32)
# takes 15 mins
distances_matrix = metrics.distances_matrix(word_embeddings, metrics.soft_cosine_similarity)
np_distances_matrix = np.array(distances_matrix)
np.save(os.path.join(os.getcwd(), "data/gensim-matrix-cossim.npy"), np_distances_matrix)
np_distances_matrix = np.load(os.path.join(os.getcwd(), "data/gensim-matrix-cossim.npy"))

np_soft_cos_sim_matrix = np.array(metrics.distances_matrix(word_embeddings, metrics.soft_cosine_similarity))
exit(12)

word2find = "ksiądz"
word_synonyms = metrics.top_most_similar_words(word2find, np_soft_cos_sim_matrix, 6)
print(f"Top 12 most similar words to `{word2find}` are in order: {' '.join(word_synonyms)}")

print(metrics.distances("ksiądz", metrics.np_soft_cos_sim_matrix, word2row, row2word, metrics.euclidean_distance)["bernardyn"])

exit(12)

top_word, top_list = metrics.find_word_with_the_closest_words_from_all(np_distances_matrix)
print(f"Word with most similar words is {top_word} and the words are: {' '.join([str(x) for x in top_list])}")
exit(76)

