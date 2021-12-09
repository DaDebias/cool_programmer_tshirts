#from we import *
#E = WordEmbedding("/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin")
import numpy as np

from gensim.models.keyedvectors import KeyedVectors
k_model_w2v = KeyedVectors.load_word2vec_format('/work/dagw_wordembeddings/word2vec_model/DAGW-model.bin', binary=True) 

# Word list
Wl = ['mand', 'kvinde', 'rig', 'fattig', 'dronning',
      'konge', 'skuespillerinde', 'skuespiller']
Wv = []
for i in range(len(Wl)):
    # Embeddings
    Wv.append(k_model_w2v[Wl[i]])
# To-be basis
b1 = (Wv[1]-Wv[0])
b2 = (Wv[3]-Wv[2])
# Get pseudo-inverse matrix
W = np.array(Wv)
B = np.array([b1,b2])
Bi = np.linalg.pinv(B.T)
# Project all the words
Wp = np.matmul(Bi,W.T)
Wp = (Wp.T-[Wp[0,2],Wp[1,0]]).T

import matplotlib.pyplot as plt
import pandas as pd
#Wp = Wp.T
df = pd.DataFrame(Wp, index=Wl, columns=['x', 'y']) # create a dataframe for plotting
print(df)
# create a plot object
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# add point for the words
ax.scatter(df['x'], df['y'])

# add word label to each point
for word, pos in df.iterrows():
    ax.annotate(word, pos)

#print(ax)

#words_to_plot = ["man", "woman", "queen", "king", "boy", "girl", "actor", "actress", "male", "female"]
#ax = plot_word_embeddings(words=words_to_plot, embedding=word_emb)
ax.plot()
