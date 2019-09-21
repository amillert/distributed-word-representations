import os
import re
from smart_open import smart_open as open
import numpy as np
from torch.utils.data import Dataset
from collections import Counter, defaultdict


class TadziuDataset(Dataset):
    def __init__(self, args):
      # MAIN_DIR = os.getcwd()
      MAIN_DIR = "/home/amillert/private/distributed-word-representations/"
      ELLIPSIS, QUOTE, MINUS = chr(8230), chr(187), chr(8212)
      text = open(os.path.join(MAIN_DIR, "data/pan-tadziu.txt"), encoding="utf-8").read().replace(ELLIPSIS, ".").replace(QUOTE, "").replace(MINUS, "")
      pol_stop_words = open(os.path.join(MAIN_DIR, "data/polish-stopwords.txt"), encoding="utf-8").read().split("\n")

      REGEX_DOC_SPLIT = "(Księga)[a-z ąćęłóźż]*[\n]{4,}"
      DOC_SPLITTER = "`~`"
      docs = [' '.join(' '.join(x.split("\n\n")[1:]).split("    ")[1:]).replace("\n", " ") for x in re.sub(REGEX_DOC_SPLIT, DOC_SPLITTER, text).split(DOC_SPLITTER) if x != ""]

      docs2sentences = [re.sub("[.!?]", ".", doc).strip().split(".") for doc in docs]
      docs2sentences = [[re.sub("[,;:*(){}\-\/]", "", sentence.strip()) for sentence in doc if sentence != ""] for doc in docs2sentences]
      docs2sentences2tokens = [[[word.lower() for word in sentence.split() if word.lower() not in pol_stop_words] for sentence in doc] for doc in docs2sentences]

      tokens = [word for sentence in docs2sentences2tokens for words in sentence for word in words]
      self.vocab = [word for word in sorted(list(set(tokens)))]

      FAKE_WORD = "fake"
      self.vocab.append(FAKE_WORD)
      self.vocab_size = len(self.vocab)
      self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
      self.idx2word = self.invert_word2idx()
      self.FAKE = self.word2idx[FAKE_WORD]
      window = 6

      # per sentence approach
      # tcp = []
      # for doc in docs2sentences2tokens:
      #   for sentence in doc:
      #     for center_idx in range(len(sentence)):
      #       center_context_words = []
      #       for context in range(-window, window + 1):
      #       # for context in range(-window // 2, window // 2 + 1):
      #         context_idx = center_idx + context
      #         if context_idx < 0 or context_idx >= len(sentence):
      #           center_context_words.append(self.FAKE)
      #         elif context_idx != center_idx:
      #           center_context_words.append(self.word2idx[sentence[context_idx]])
      #       tcp.append(tuple([self.word2idx[sentence[center_idx]], np.array(center_context_words)]))

      docs2tokens = [[word for sentence in doc for word in sentence] for doc in docs2sentences2tokens]
      tcp = []
      for tokens in docs2tokens:
          for center_idx in range(len(tokens)):
            center_context_words = []
            for context in range(-window, window + 1):
              context_idx = center_idx + context
              if context_idx < 0 or context_idx >= len(tokens):
                center_context_words.append(self.FAKE)
              elif context_idx != center_idx:
                center_context_words.append(self.word2idx[tokens[context_idx]])
            tcp.append(tuple([self.word2idx[tokens[center_idx]], np.array(center_context_words)]))
      
      self.X, self.y = zip(*tcp)
      assert len(self.X) == len(self.y)
      self.len = len(self.X)

      for i, x in enumerate(tcp):
        if i < 5: print(x)
        else: break

    def invert_word2idx(self):
      return {v: k for k, v in self.word2idx.items()}

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.len

