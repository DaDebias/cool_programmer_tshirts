import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd

model = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 
#k_model_w2v = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/debiased_model.bin', binary=True) 
Wl = ['ude', 'hjemme','rig', 'fattig', 'dronning', 'konge', 'skuespillerinde', 'skuespiller']
from numpy import loadtxt
b2 = np.loadtxt('/work/Exam/dk-weat/output/neutral_specific_difference.csv', delimiter=',')
 xx = ['mand', 'kvinde']

def plot_professions(embedding, wordlist, x_axis, y_axis):

    vectors = []

    words = x_axis + wordlist
    
    for i in range(len(words)):
        # Embeddings
        vectors.append(embedding[words[i]])
    # To-be basis

    x = (vectors[1]-vectors[0])
    y = (vectors[3]-vectors[2])
    
    # Get pseudo-inverse matrix
    W = np.array(vectors)
    B = np.array([x,y])
    Bi = np.linalg.pinv(B.T)
    
    # Project all the words
    Wp = np.matmul(Bi,W.T)
    Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T
    Wp = Wp.T
    
    df = pd.DataFrame(Wp, index=words, columns=['x', 'y']) # create a dataframe for plotting
    
    # create a plot object
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # add point for the words
    ax.scatter(df['x'], df['y'])
    ax.set_ylabel('genderedness')
    ax.set_xlabel('difference of mand-kvinde')
    # add word label to each point
    for w, pos in df.iterrows():
        ax.annotate(w, pos)

    return ax.plot()

plot_professions(model, Wl,xx, b2)