import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import json
from numpy import loadtxt
if sys.version_info[0] < 3:
    import io
    open = io.open
plt.style.use("seaborn")


model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 
debiased_model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/debiased_model.bin', binary=True)

wordlist = ['revisor', 'sygeplejerske','prinsesse', 'dronning','skuespiller', 'skuespillerinde', 'advokat', 'hjælper',  'antropolog', 'arkæolog', 'arkitekt', 'kunstner',  'morder', 'astronaut', 'astronom', 'atlet',  'forfatter', 'bager', 'ballerina', 'fodboldspiller', 'bankmand', 'barber', 'baron', 'bartender', 'biolog', 'biskop', 'præst']
x_ax = ['kvinde', 'mand']
y_ax = np.loadtxt('/work/Exam/dk-weat/output/neutral_specific_difference.csv', delimiter=',')

def plot_professions(embedding, wordlist, x_axis, y_axis):

    wordlist = x_axis + wordlist
    vectors = []


    for i in range(len(wordlist)):
        # Embeddings
        vectors.append(embedding[wordlist[i]])
    # To-be basis
    x = (vectors[1]-vectors[0])
    #flipped
    y = np.flipud(y_axis)
    #reversed y axis
    #y = (vectors[3]-vectors[2]) #np.loadtxt('/work/Exam/dk-weat/output/neutral_specific_difference.csv', delimiter=',')

    # Get pseudo-inverse matrix
    W = np.array(vectors)
    B = np.array([x,y])
    Bi = np.linalg.pinv(B.T)

    print(B.shape)
    print(W.T.shape)
    print(Bi.shape)

    # Project all the words
    Wp = np.matmul(Bi,W.T)
    Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T
    print(Wp.shape)

    #PLOT
    plt.figure(figsize=(12,7))
    plt.axvline()
    plt.axhline()
    plt.title(label="Professions",
            fontsize=30,
            color="black")
    plt.xlim([-1, 1])
    plt.scatter(Wp[0,:], Wp[1,:])
    #rX = max(Wp[0,:])-min(Wp[0,:])
    rX = max(Wp[0,:])-min(Wp[0,:])
    rY = max(Wp[1,:])-min(Wp[1,:])
    eps = 0.005

    #plt.ylim([-0.0005, 0.20])
    for i, txt in enumerate(wordlist):
        #plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rX*eps))
        plt.annotate(txt, (Wp[0,i]+rX*eps, Wp[1,i]+rY*eps))
    plt.show()

plot_professions(model, wordlist, x_ax, y_ax)
plot_professions(debiased_model, wordlist, x_ax, y_ax)