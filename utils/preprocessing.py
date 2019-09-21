import re
import os
import copy
import numpy as np
from smart_open import smart_open
from collections import Counter


def invert_dict(dic):
  return {v: k for k, v in dic.items()}

def preprocess_pan_tadeusz():
  MAIN_DIR = os.getcwd()
  ELLIPSIS = chr(8230)
  LEFTQUOTE = chr(171)
  RIGHTQUOTE = chr(187)
  MINUS = chr(8212)
  text = open(os.path.join(MAIN_DIR, "data/pan-tadziu.txt"), encoding="utf-8").read().replace(ELLIPSIS, ".").replace(LEFTQUOTE, "").replace(RIGHTQUOTE, "").replace(MINUS, "")
    
  REGEX_DOC_SPLIT = "(Księga)[a-z ąćęłóźż]*[\n]{4,}"
  DOC_SPLITTER = "`~`"
  text_splitted_to_docs = [' '.join(' '.join(x.split("\n\n")[1:]).split("    ")[1:]).replace("\n", " ") for x in re.sub(REGEX_DOC_SPLIT, DOC_SPLITTER, text).split(DOC_SPLITTER) if x != ""]
  return text_splitted_to_docs

def tokenize_docs_sentences(text_splitted_to_docs):
  docs2sentences = [re.sub("[.!?]", ".", doc).strip().split(".") for doc in text_splitted_to_docs]
  docs2sentences = [[re.sub("[,;:*(){}\-\/]", "", sentence.strip()) for sentence in doc if sentence != ""] for doc in docs2sentences]
  # [[[slowo A], [slowo B], [slowo C]]:
  polish_stop_words = open(os.path.join(os.getcwd(), "data/polish-stopwords.txt"), encoding="utf-8").read().split("\n")
  docs2sentences2tokens = [[[word.lower() for word in sentence.split() if word.lower() not in polish_stop_words] for sentence in doc] for doc in docs2sentences]
  return docs2sentences2tokens

def target_context_pairs_operations(docs2sentences2tokens):
  tokens = [word for doc in docs2sentences2tokens for sentence in doc for word in sentence]
  print(f"Amount of tokens: {len(tokens)}")
  vocab = sorted(list(set(tokens)))
  print(f"Vocab size: {len(vocab)}")
  word2idx = {word: idx for idx, word in enumerate(vocab)}
  idx2word = invert_dict(word2idx)

  window = 4
  thresh = 1

  docs2tokens = [[word for sentence in doc for word in sentence] for doc in docs2sentences2tokens]
  # TOKENS PER SENTENCE IN DOCUMENT:
  all_tcp = []
  for tokens in docs2tokens:
    for center_idx in range(len(tokens)):
      for context in range(-window, window + 1):
      # for context in range(-window // 2, window // 2 + 1):
        context_idx = center_idx + context
        if context_idx < 0 or context_idx >= len(tokens):
          all_tcp.append(tuple([center_idx, -1]))
        elif context_idx != center_idx:
          all_tcp.append(tuple([word2idx[tokens[center_idx]], word2idx[tokens[context_idx]]]))

  # TOKENS PER DOCUMENT:
  # all_tcp = []
  # for doc in docs2sentences2tokens:
  #   for sentence in doc:
  #     for center_idx in range(len(sentence)):
  #       for context in range(-window // 2, window // 2 + 1):
  #         context_idx = center_idx + context
  #         if context_idx < 0 or context_idx >= len(sentence):
  #           all_tcp.append(tuple([center_idx, -1]))
  #         elif context_idx != center_idx:
  #           all_tcp.append(tuple([word2idx[sentence[center_idx]], word2idx[sentence[context_idx]]]))
  
  # this representation allows ease of saving to a map
  all_tcp_strings = [f"{pair[0]} {pair[1]}" for pair in all_tcp]

  # dict with count ocurrances of target context pairs in the text
  pair2count = Counter(all_tcp_strings)

  # TESTING PAIR2COUNT FOR `KSIĄDZ ROBAK`
  # for k, v in pair2count.items():
  #   t, c = k.split(" ")
  #   if int(t) == word2idx["ksiądz"] and int(c) == word2idx["robak"]:
  #     print(k, pair2count[k])
  # exit(30)
  # pprint(sorted(pair2count.items(), key=lambda l: -l[1])[:10])

  # filtering out pairs which occur too rarely
  tcp_strings_filtered = [tcp for tcp in all_tcp_strings if pair2count[tcp] > thresh]
  print(f"Len of tcp strings filtered: {len(tcp_strings_filtered)}")
  
  tcp_idx = [[int(x[0]), int(x[1])] for x in [pair.split(" ") for pair in tcp_strings_filtered]]
  print(f"Len of tcp idx filtered: {len(tcp_idx)}")
  vocab_idx = sorted(list(set([idx for x in tcp_idx for idx in x if idx != -1])))
  print(f"New vocab filtered size: {len(vocab_idx)}")
  # print(len(word2idx))
  word2idx = {idx2word[idx]: idx for idx in vocab_idx}
  # print(len(word2idx))
  return word2idx, tcp_idx, vocab_idx, pair2count

def get_word_coocurrence_dict(word2idx, tcp_idx, vocab_idx, pair2count):
  word_row = {word_idx: 0 for word_idx in vocab_idx}
  coocurrence_dict = {word_idx: copy.deepcopy(word_row) for word_idx in vocab_idx}
  
  row2idx = {row: idx for idx, row in zip(vocab_idx, range(len(vocab_idx)))}
  idx2row = invert_dict(row2idx)
  
  fake = -1
  
  for target, context in tcp_idx:
    if target != fake != context:
      if pair2count[f"{target} {context}"] > 0:
        coocurrence_dict[target][context] = pair2count[f"{target} {context}"]
  return coocurrence_dict

def get_coocurrence_matrix(vocab_idx, coocurrence_dict):
  coocurrence_matrix_np = np.array([list(coocurrence_dict[idx].values()) for idx in vocab_idx])
  path = "/home/amillert/private/distributed-word-representations/data/tmp.npy"
  # np.save(path, coocurance_matrix_np)
  coocurrence_matrix = np.load(path)
  return coocurrence_matrix_np

def show_row(word, vector, row2word):
  joined = ', '.join([': '.join(['`' + row2word[i] + '`', str(value)]) for i, value in zip([x for x in range(len(vector))], vector) if value > 0])
  print(f"Values found for `{word}` are -> {joined}")

def preview_coocurance_matrix(coocurrence_matrix_np, row2word, _word=None):
  if _word:
    row = invert_dict(row2word)[_word]
    coocurrence_matrix_np = np.array(coocurrence_matrix_np[row]).reshape(1,-1)
  for row, vector in enumerate(coocurrence_matrix_np):
    word = _word if _word else row2word[row]
    show_row(word, vector, row2word)

