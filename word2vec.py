import gensim.models
import gensim.downloader as api
from datasets import load_from_disk, concatenate_datasets
import numpy as np
from numpy.linalg import norm

train_new_model = True  # set to False if you want load a trained model instead of training a new one
model_name = "w2v_model_imdb"  # use this variable to name your models

if train_new_model:
    # Make sure to run "imdb_preprocessing.py" first to get the correctly preprocessed dataset!
    corpus = concatenate_datasets([load_from_disk("imdb_preprocessed_train"),
                                   load_from_disk("imdb_preprocessed_test")]
                                  )["text"]
    model = gensim.models.Word2Vec(sentences=corpus,
                                   sg=1)
    model.save(f"{model_name}.model")
else:
    model = gensim.models.Word2Vec.load(f"{model_name}.model")

words_to_check = ["good", "great", "worst", "weakest"]
pairs = [(words_to_check[i], words_to_check[j]) for i in range(len(words_to_check)) for j in range(i + 1, len(words_to_check))]
print("Cosine similarities using w2v-model trained on IMDB:")
for (w1, w2) in pairs:
    print(f"\tpair - ({w1}, {w2}): {model.wv.similarity(w1, w2):.2f}")

wv = api.load('word2vec-google-news-300')
print("Cosine similarities using w2v-model trained on the google news dataset:")
for (w1, w2) in pairs:
    print(f"\tpair - ({w1}, {w2}): {wv.similarity(w1, w2):.2f}")