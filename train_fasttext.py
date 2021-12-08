from gensim.models import FastText
from gensim.utils import tokenize

import os
import random
import time

import spacy
from typing import Iterable

nlp = spacy.load("da_core_news_lg", disable=["ner", "pos", "morphologizer", "attribute_ruler", "lemmatizer"])
nlp.max_length = 4000000

dagw_sektioner = "/data/DAGW/dagw/sektioner"
fasttext_path = "model/fasttext.model"

def shuffled_filepaths():
    sections = os.listdir(dagw_sektioner)

    filepaths = {}
    for p in sections:
        subpath = os.path.join(dagw_sektioner, p)
        filepaths[p] = [
            os.path.join(subpath, p)
            for p in os.listdir(subpath)
            if p != "LICENSE" and not p.endswith(".jsonl")
        ]
    def handle_subdir(section):
        return  [os.path.join(filepaths[section][0], p)
            for p in os.listdir(filepaths[section][0])]

    filepaths["twfv19"] = handle_subdir("twfv19")
    filepaths["datwitter"] = handle_subdir("datwitter")
    # flatten to list
    files = [file for key in filepaths for file in filepaths[key]]
    random.shuffle(files)
    return files


def text_gen(filepaths=shuffled_filepaths()):
    for i, file in enumerate(filepaths):
        if i % 10000 == 0:
            print("\t", i, "/", len(filepaths))
        with open(file, "r") as f:
            text = f.read()
            if len(text) < nlp.max_length:
                yield text
            else: 
                continue

def tokenize_texts(texts: Iterable[str] = text_gen()):
    docs = nlp.pipe(texts, batch_size=8, n_process=10)
    for doc in docs:
        for sent in doc.sents:
            yield [t.text for t in sent]


   
def train_fasttext():
    model = FastText(vector_size=500, window=5, min_count=5, workers=10)
    print(f"[INFO] Building vocab...")
    t0 = time.time()
    model.build_vocab(corpus_iterable=tokenize_texts())
    print(f"[INFO] Done in {time.time() - t0}")
    total_examples = model.corpus_count
    print(f"[INFO] Training model...")
    model.train(corpus_iterable=tokenize_texts(), total_examples=total_examples, epochs=1)
    print(f"[INFO] Done in {time.time() - t0}")
    print(f"[INFO] Saving.. ")
    model.save(fasttext_path)



if __name__ == "__main__":
    train_fasttext()