from __future__ import division

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
import numpy as np
import pandas as pd

import csv, sklearn
from sklearn import metrics

from numpy import bincount, log, sqrt
from scipy.sparse import coo_matrix, csr_matrix

import scipy.sparse
from sparsesvd import sparsesvd

import math as mt



#EPOCHS = 10

batch_size = 512
EPOCHS = 100


def bm25_weight(X, K1=100, B=0.8):
    """ Weighs each row of a sparse matrix X  by BM25 weighting """
    # calculate idf per term (user)
    X = coo_matrix(X)

    N = float(X.shape[0])
    idf = log(N / (1 + bincount(X.col)))

    # calculate length_norm per document (artist)
    row_sums = np.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X



def tfidf_weight(X):
    """ Weights a Sparse Matrix by TF-IDF Weighted """
    X = coo_matrix(X)

    # calculate IDF
    N = float(X.shape[0])
    idf = log(N / (1 + bincount(X.col)))

    # apply TF-IDF adjustment
    X.data = sqrt(X.data) * idf[X.col]
    return X



def trans2vect( df,uid,pid,rate,top ):
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize
    
    #sparse matrix with product in rows and users in columns
    df=df[df['song_id'].isin(top.index)]
    user_u = list(df[uid].unique())
    song_u = list(top.index)
    col = df[uid].astype('category', categories=user_u).cat.codes
    row = df[pid].astype('category', categories=song_u).cat.codes
    songrating = csr_matrix((df[df[pid].isin(song_u)][rate].tolist(), (row,col)), shape=(len(song_u),len(user_u)))
    
    #normalize "l1"
    matrixNormli = normalize(songrating, norm='l1', axis=0)
    #normalize "bm25"
    matrixNormbm25 = bm25_weight(songrating)
    #normalize "tdidf"
    matrixNormtdidf = tfidf_weight(songrating)
    return matrixNormli, matrixNormbm25, matrixNormtdidf



def computeSVD(urm, K):
	U, s, Vt = sparsesvd(urm, K)

	dim = (len(s), len(s))
	S = np.zeros(dim, dtype=np.float32)
	for i in range(0, len(s)):
		S[i,i] = mt.sqrt(s[i])

	U = csr_matrix(np.transpose(U), dtype=np.float32)
	S = csr_matrix(S, dtype=np.float32)
	Vt = csr_matrix(Vt, dtype=np.float32)

	return U, S, Vt	




if __name__ == "__main__":

    data_path = 'input/'
    train = pd.read_csv(data_path + 'train.csv')
    songs = pd.read_csv(data_path + 'songs.csv')
    #print(songs.head())

    train['rating']=1
    #train the skippers

    #top target skipped songs, needs rating to sum
    topsongs=train.groupby(by=['song_id'])['rating'].sum()

    ratingsli, ratingsbm25, ratingstdidf = trans2vect(train,'msno','song_id','rating', topsongs)

    #Normalize L1
 
    smat = scipy.sparse.csc_matrix(ratingsli) 

    Uli, Sli, Vtli = computeSVD(smat, 100 )  #Uli, (359966, 100) Sli, (100, 30755) Vtli , (100, 100)

    #Normalize BM25
 
    smat = scipy.sparse.csc_matrix(ratingsbm25) 

    Ubm25, Sbm25, Vtbm25 = computeSVD(smat, 100 ) #Ubm25, (359966, 100) Sbm25, (100, 30755) Vtbm25, (100, 100)

    #Normalize td-idf
 
    smat = scipy.sparse.csc_matrix(ratingstdidf) 

    Utf, Stf, Vttf = computeSVD(smat, 100 ) #Utf, (359966, 100) Stf, (100, 30755) Vttf, (100, 100)

    #concatenate the vector
    X = np.hstack(( Uli.todense(), Ubm25.todense(), Utf.todense() ))
    y = train['target']

    print "Size of X: {}, and size of y: {}".format ( X.shape, y.shape )





