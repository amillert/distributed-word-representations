import re
import os
import copy
import time
import numpy as np
from smart_open import smart_open as open
from pprint import pprint
from collections import Counter

from torch.utils.data import Dataset

from devmood.nn.learning.nn_learner import learn


# MAIN_DIR = os.getcwd()
# ELLIPSIS = chr(8230)
# QUOTE = chr(187)
# MINUS = chr(8212)
# text = open(os.path.join(MAIN_DIR, "data/pan-tadziu.txt"), encoding="utf-8").read().replace(ELLIPSIS, ".").replace(QUOTE, "").replace(MINUS, "")
# pol_stop_words = open(os.path.join(os.getcwd(), "data/polish-stopwords.txt"), encoding="utf-8").read().split("\n")
# 
# REGEX_DOC_SPLIT = "(Księga)[a-z ąćęłóźż]*[\n]{4,}"
# DOC_SPLITTER = "`~`"
# docs = [' '.join(' '.join(x.split("\n\n")[1:]).split("    ")[1:]).replace("\n", " ") for x in re.sub(REGEX_DOC_SPLIT, DOC_SPLITTER, text).split(DOC_SPLITTER) if x != ""]
# 
# docs2sentences = [re.sub("[.!?]", ".", doc).strip().split(".") for doc in docs]
# docs2sentences = [[re.sub("[,;:*(){}\-\/]", "", sentence.strip()) for sentence in doc if sentence != ""] for doc in docs2sentences]
# docs2sentences2tokens = [[[word.lower() for word in sentence.split() if word.lower() not in pol_stop_words] for sentence in doc] for doc in docs2sentences]
# 
# fake = -1
# window = 4
# thresh = 1
#  
# def invert_dict(dic):
#   return {v: k for k, v in dic.items()}
# 
# def get_all_stuff(docs2sentences2tokens):
#   tokens = [word for doc in docs2sentences2tokens for sentence in doc for word in sentence]
#   print(f"Amount of tokens: {len(tokens)}")
#   vocab = sorted(list(set(tokens)))
#   print(f"Vocab size: {len(vocab)}")
#   word2idx = {word: idx for idx, word in enumerate(vocab)}
#   idx2word = invert_dict(word2idx)
#   
#   all_tcp = []
#   for doc in docs2sentences2tokens:
#     for sentence in doc:
#       for center_idx in range(len(sentence)):
#         for context in range(-window // 2, window // 2 + 1):
#           context_idx = center_idx + context
#           if context_idx < 0 or context_idx >= len(sentence):
#             all_tcp.append(tuple([center_idx, fake]))
#           elif context_idx != center_idx:
#             all_tcp.append(tuple([word2idx[sentence[center_idx]], word2idx[sentence[context_idx]]]))
#   
#   all_tcp_strings = [f"{pair[0]} {pair[1]}" for pair in all_tcp]
#   print(f"Len of all tcps: {len(all_tcp)}")
# 
#   pair2count = Counter(all_tcp_strings)
#   # pprint(sorted(pair2count.items(), key=lambda l: -l[1])[:10])
# 
#   tcp_strings_filtered = [tcp for tcp in all_tcp_strings if pair2count[tcp] > thresh]
#   print(f"Len of tcp strings filtered: {len(tcp_strings_filtered)}")
#   
#   tcp_idx = [[int(x[0]), int(x[1])] for x in [pair.split(" ") for pair in tcp_strings_filtered]]
#   print(f"Len of tcp idx filtered: {len(tcp_idx)}")
#   vocab_idx = sorted(list(set([idx for x in tcp_idx for idx in x if idx != fake])))
#   print(f"New vocab filtered size: {len(vocab_idx)}")
#   word2idx = {idx2word[idx]: idx for idx in vocab_idx}
# 
#   return word2idx, tcp_idx, vocab_idx, pair2count
# 
# word2idx, tcp_idx, vocab_idx, pair2count = get_all_stuff(docs2sentences2tokens)
# idx2word = invert_dict(word2idx)
# 
# inner = {word_idx: 0 for word_idx in vocab_idx}
# alls = {word_idx: copy.deepcopy(inner) for word_idx in vocab_idx}
# 
# # TODO: make suer the keys are alway in good order; think about how to check if values are good (in pairs should have each other?)
# # matrix_np = np.zeros((len(vocab_idx), len(vocab_idx)))
# 
# row2idx = {row: idx for idx, row in zip(vocab_idx, range(len(vocab_idx)))}
# idx2row = invert_dict(row2idx)
# 
# for target, context in tcp_idx:
#   if target != fake != context:
#     if pair2count[f"{target} {context}"] > 0:
#       alls[target][context] = pair2count[f"{target} {context}"]
# 
# path = "/home/amillert/private/distributed-word-representations/data/tmp.npy"
# matrix_np = np.array([list(alls[idx].values()) for idx in vocab_idx])
# np.save(path, matrix_np)
# matrix_np = np.load(path)
# 
# Preview of the matrix:
# for i, x in enumerate(matrix_np):
#   print(f"Values found for the word {idx2word[row2idx[i]]} are {' '.join(set([str(xi) for xi in x]))}")

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

triangle = np.array([[ 2.0,  4.0], [10.0, 15.0], [14.0, 10.0]])

# think of counting the area of the triangles given those metrics
# for distance_metric in (euclidean_distance, cosine_distance, jaccard_distance, jensen_shannon_distance, dice_coefficient):
#   #format = {"name": distance_metric.__name__,  "AB": distance_metric(triangle[0], triangle[1]), "BC": distance_metric(triangle[1], triangle[2])}
#   print(f"{distance_metric.__name__}")
#   print(f"|AB| = {distance_metric(triangle[0], triangle[1])}")
#   print(f"|BC| = {distance_metric(triangle[1], triangle[2])}")
#   print(f"|AC| = {distance_metric(triangle[0], triangle[2])}", end="\n")

# def neighbours(word, matrix, rownames, distance=euclidean_distance):
#   palabra = matrix[rownames.index(word)]
#   distances = [(rownames[i], distance(palabra, matrix[i])) for i in xrange(len(mat))]
#   return sorted(distances, key=lambda l: l[1])
#   # return sorted(distances, key=itemgetter(1), reverse=False)
# 
# neighbours("ksiądz", matrix_np, rownames=ww[1])
args = {"batch_size": 128, "dims": 50, "eta": 0.01, "epochs": 100, "input": "Pan Tadeusz"}
learn(args)

