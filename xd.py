import re
import os
from smart_open import smart_open as open
from pprint import pprint
from collections import Counter


MAIN_DIR = os.getcwd()
ELLIPSIS = chr(8230)
QUOTE = chr(187)
MINUS = chr(8212)
text = open(os.path.join(MAIN_DIR, "data/pan-tadziu.txt"), encoding="utf-8").read().replace(ELLIPSIS, ".").replace(QUOTE, "").replace(MINUS, "")
pol_stop_words = open(os.path.join(os.getcwd(), "data/polish-stopwords.txt"), encoding="utf-8").read().split("\n")

REGEX_DOC_SPLIT = "(Księga)[a-z ąćęłóźż]*[\n]{4,}"
DOC_SPLITTER = "`~`"
docs = [' '.join(' '.join(x.split("\n\n")[1:]).split("    ")[1:]).replace("\n", " ") for x in re.sub(REGEX_DOC_SPLIT, DOC_SPLITTER, text).split(DOC_SPLITTER) if x != ""]

docs2sentences = [re.sub("[.!?]", ".", doc).strip().split(".") for doc in docs]
docs2sentences = [[re.sub("[,;:*(){}\-\/]", "", sentence.strip()) for sentence in doc if sentence != ""] for doc in docs2sentences]

def get_all_stuff(docs2sentences):
  docs2sentences2tokens = [[[word.lower() for word in sentence.split()] for sentence in doc] for doc in docs2sentences]
  tokens = [word for doc in docs2sentences2tokens for sentence in doc for word in sentence]
  print(f"Amount of tokens: {len(tokens)}")
  vocab = sorted(list(set(tokens)))
  print(f"Vocab size: {len(vocab)}")
  word2idx = {word: idx for idx, word in enumerate(vocab)}
  idx2word = {idx: word for word, idx in word2idx.items()}
  
  fake = -1
  window = 4
  
  all_tcp = []
  for doc in docs2sentences2tokens:
    doc_tcp = []
    for sentence in doc:
      tcp = []
      for center_idx in range(len(sentence)):
        for context in range(-window // 2, window // 2 + 1):
          context_idx = center_idx + context
          if context_idx < 0 or context_idx >= len(sentence):
            tcp.append(tuple([center_idx, fake]))
          elif context_idx != center_idx:
            tcp.append(tuple([word2idx[sentence[center_idx]], word2idx[sentence[context_idx]]]))
        doc_tcp.append(tcp)
    all_tcp.append(doc_tcp)
  
  print(f"Len of all tcps: {len(all_tcp)}")
  
  all_tcp_strings = [f"{pair[0]} {pair[1]}" for doc_tcp in all_tcp for tcp in doc_tcp for pair in tcp]
  pair2count = Counter(all_tcp_strings)
  word2count = Counter(tokens)
  
  print(f"Len of pair2counts: {len(pair2count.keys())}")
  print(f"Len of word2ounts: {len(word2count.keys())}")
  pprint(sorted(pair2count.items(), key=lambda l: -l[1])[:10])
  pprint(sorted(word2count.items(), key=lambda l: -l[1])[:10])
  return word2idx, idx2word, all_tcp, all_tcp_strings, pair2count, word2count, vocab

word2idx, idx2word, all_tcp, all_tcp_strings, pair2count, word2count, vocab = get_all_stuff(docs2sentences)

docs2sentences = [[' '.join([word.lower() for word in sentence.split() if word.lower() not in pol_stop_words]) for sentence in doc] for doc in docs2sentences]

print("")
print("")
print("")

word2idx, idx2word, all_tcp, all_tcp_strings, pair2count, word2count, vocab = get_all_stuff(docs2sentences)


