from __future__ import print_function, division
import argparse
import sys
import we
import json
import numpy as np
if sys.version_info[0] < 3:
    import io
    open = io.open 
"""
Hard-debias embedding

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""


def debias(E, gender_specific_words, definitional, equalize):
    
    # define gender direction

    pca = we.doPCA(definitional, E).components_[0]
    pca = we.doPCA(definitional, E).components_[0]
    gender_direction = pca.components_[0]

    # save gender direction (to print most extreme job professions)
    np.savetxt('/work/Exam/dk-weat/output/gender_direction.csv', gender_direction, delimiter=',')

    # load full genderspecific
    specific_set = set(gender_specific_words)

    # neutralize: go through entire wordembedding - remove  gender direction from words not in full gender specific
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    
    # normalize values in Embedding
    E.normalize()

    # equalize: take all equalize pairs (both in upper/lowercanse) 
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    
    # normalize Embedding
    E.normalize()



def debias_no_equalize(E, gender_specific_words, definitional, equalize):
    
    # define gender direction
    gender_direction = we.doPCA(definitional, E).components_[0]

    # save gender direction (to print most extreme job professions)
    np.savetxt('/work/Exam/dk-weat/output/gender_direction.csv', gender_direction, delimiter=',')

    # load full genderspecific
    specific_set = set(gender_specific_words)

    # neutralize: go through entire wordembedding - remove  gender direction from words not in full gender specific
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    
    # normalize values in Embedding
    E.normalize()



if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument("--embedding_filename", help="The name of the embedding")
    #parser.add_argument("--definitional_filename", help="JSON of definitional pairs")
    #parser.add_argument("--gendered_words_filename", help="File containing words not to neutralize (one per line)")
    #parser.add_argument("--equalize_filename", help="???.bin")
    #parser.add_argument("--debiased_filename", help="???.bin")

    #args = parser.parse_args()
    #print(args)
    embedding_filename = "/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin" 

    definitional_filename = "/work/Exam/cool_programmer_tshirts/data/da_definitional_pairs.json" 

    gendered_words_filename = "/work/Exam/cool_programmer_tshirts/data/gender_specific_full.json" 

    equalize_filename = "/work/Exam/cool_programmer_tshirts/data/da_equalize_pairs.json"
    
    debiased_filename = "debiased_model.bin"



    # load words for creating gender direction
    with open(definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    # load word paris to equalize
    with open(equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    # load full gender specific
    with open(gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    # load word embedding
    E = we.WordEmbedding(embedding_filename)

    # debias word embedding
    print("Debiasing...")
    
    # debias model: neutralize
    debias_no_equalize(E, gender_specific_words, defs, equalize_pairs)

    # debias model: neutralize + equalize
    #debias(E, gender_specific_words, defs, equalize_pairs)


    # save debased
    print("Saving to file...")
    if embedding_filename[-4:] == debiased_filename[-4:] == ".bin":
        E.save_w2v(f"no_eq_{debiased_filename}")
    else:
        E.save(f"no_eq_{debiased_filename}")

    print("\n\nDone!\n")

