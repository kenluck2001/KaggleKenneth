from __future__ import division
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import datetime as dt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.svm import NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import f1_score
'''
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
'''
from gplearn.genetic import SymbolicRegressor

from pyearth import Earth

def prepossMLP ( X_train, X_test ):

    # pre-processing: divide by max and substract mean
    scale = np.max(X_train)
    X_train /= scale
    X_test /= scale

    mean = np.std(X_train)
    X_train -= mean
    X_test -= mean
    return X_train, X_test


def mlpClassifier( input_dim, nb_classes ):

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model



def AUC(y, pred):
    auc = roc_auc_score(y, pred)
    return  auc 



def crossValidation (model, X, y, nfolds = 5 ):
    nrow, ncol = X.shape
    meanSize = nrow // nfolds

    y = y.to_frame()


    score = 0.0

    meanlist = []

    dataindex = []

    for ind in range (0, nfolds):
        start, end = ind * meanSize, (ind+1) * meanSize
        dataindex.append ( (start, end) )


    for ind, curind in enumerate (dataindex):
        start, end = curind
        nextind = ind + 1
    
        Xcur, ycur = X.iloc[start:end, :], y[start:end]

        model.fit(Xcur, ycur)

        #get test
        if nextind < len(dataindex):

            tstart, tend = dataindex[nextind]
            Xtcur, ytcur = X.iloc[tstart:tend, :], y.iloc[tstart:tend, :]

            ypred = model.predict(Xtcur)


            ypred = ypred.reshape( (1, len(ypred) ) )

            #ytcur = ytcur.reshape( (1, len(ytcur) ) ) 

            ypred = ypred.T

            ytcur = ytcur.values

            #print ytcur.shape, ypred.shape
            #print type (ytcur), type( ypred ) 
            #nmlist = ytcur.tolist()


            #score = AUC( ytcur, ypred )

            score = f1_score( ytcur.astype(int), ypred.astype(int), average='macro')  

        meanlist.append ( score )
    return sum (meanlist) / len (meanlist)



def crossValidationMLP (model, X, y, nfolds = 5 ):
    nrow, ncol = X.shape
    meanSize = nrow // nfolds

    y = y.to_frame()

    #scale = np.max( X )

    #X /= scale

    score = 0.0

    meanlist = []

    dataindex = []

    for ind in range (0, nfolds):
        start, end = ind * meanSize, (ind+1) * meanSize
        dataindex.append ( (start, end) )


    for ind, curind in enumerate (dataindex):
        start, end = curind
        nextind = ind + 1
    
        Xcur, ycur = X.iloc[start:end, :], y[start:end]



        model.fit(Xcur, ycur, nb_epoch=10, batch_size=16, validation_split=0.1)


        #get test
        if nextind < len(dataindex):

            tstart, tend = dataindex[nextind]
            Xtcur, ytcur = X.iloc[tstart:tend, :], y.iloc[tstart:tend, :]

            ypred = model.predict(Xtcur)


            ypred = ypred.reshape( (1, len(ypred) ) )

            #ytcur = ytcur.reshape( (1, len(ytcur) ) ) 

            ypred = ypred.T

            ytcur = ytcur.values

            #print ytcur.shape, ypred.shape
            #print type (ytcur), type( ypred ) 
            #nmlist = ytcur.tolist()


            #score = AUC( ytcur, ypred )

            score = f1_score( ytcur.astype(int), ypred.astype(int), average='macro')  

        meanlist.append ( score )
    return sum (meanlist) / len (meanlist)




def label(df,var):
    for i in var:
        df[i]= le.fit_transform(df[i])


def date_feature(df):
    var = ['registration_init_time','expiration_date']
    k = ['reg','exp']
    df['sub_duration'] = (df[var[1]] - df[var[0]]).dt.days
    for i ,j in zip(var,k):
        df[j+'_day'] = df[i].dt.day
        df[j+'_weekday'] = df[i].dt.weekday
        df[j+'_week'] = df[i].dt.week
        df[j+'_month'] = df[i].dt.month
        df[j+'_year'] =df[i].dt.year



seed = 23
#Load data set

data_path = 'input/'
               
train = pd.read_csv(data_path + 'train.csv',dtype=({'msno':'category','song_id':'category', 'source_system_tab':'category',
                                        'source_screen_name':'category','source_type':'category','target':'category'}))
test = pd.read_csv(data_path + 'test.csv',dtype=({'msno':'category','song_id':'category', 'source_system_tab':'category',
                                        'source_screen_name':'category','source_type':'category'}))
members = pd.read_csv(data_path + 'members.csv',parse_dates=['registration_init_time','expiration_date'],dtype=({'msno':'category','gender':'category'}))

songs = pd.read_csv(data_path + 'songs.csv',dtype=({'song_id':'category','genre_ids':'category','artist_name':'category',
                                        'composer':'category','lyricist':'category','language':'category'}))


df_train = train.merge(members,how='left',on='msno')
df_test = test.merge(members,how='left',on='msno')
df_train = df_train.merge(songs,how='left',on='song_id')
df_test = df_test.merge(songs, how='left',on='song_id')


del train,test,members,songs

cat = ['source_system_tab','source_screen_name','source_type', 'gender',
       'genre_ids','artist_name','composer','lyricist','song_length','language']

def missing(df,var):
    for i in var:
        df[i].fillna(df[i].mode()[0], inplace=True)

missing(df_train,cat)
missing(df_test,cat)


date_feature(df_train)
date_feature(df_test)


le = LabelEncoder()
cat = ['msno', 'song_id', 'source_system_tab', 'source_screen_name',
       'source_type','gender','genre_ids','artist_name','composer',
       'lyricist']


label(df_train,cat)
label(df_test,cat)




"""
#split data set 
X = df_train.drop(['target','registration_init_time', 'expiration_date'],axis=1)
y = df_train['target']
x_test = df_test.drop(['id','registration_init_time', 'expiration_date'],axis=1)
"""

lstSet = set( ['target','registration_init_time', 'expiration_date'] )

train_cols = set (df_train.columns) - lstSet
train_cols = list (train_cols )

X = df_train[train_cols]
y = df_train['target'].apply(pd.to_numeric, errors='coerce')

lstSet = set( ['id','registration_init_time', 'expiration_date'] )

test_cols = set (df_test.columns) - lstSet
test_cols = list ( test_cols )
x_test = df_test[test_cols]

"""
from catboost import CatBoostClassifier

dataset = numpy.array([[1,4,5,6],[4,5,6,7],[30,40,50,60],[20,15,85,60]])
train_labels = [1.2,3.4,9.5,24.5]
model = CatBoostClassifier(learning_rate=1, depth=6, loss_function='RMSE')
fit_model = model.fit(dataset, train_labels)

"""


if __name__ == "__main__":
    """
    xtr,xvl,ytr,yvl = train_test_split(X,y,test_size=0.3,random_state=seed)
    del X, y

    #Model
    lr = LogisticRegression(max_iter=100,random_state=seed)
    lr.fit(xtr,ytr)
    pred = lr.predict(xvl)

    model = LogisticRegression(max_iter=100,random_state=seed)

    #Submission result
    y_pred = lr.predict(x_test)
    submit = pd.DataFrame({'id':df_test['id'],'target':y_pred})
    submit.to_csv('kk_target.csv',index=False)



    print "====================================="
    lr = LogisticRegression(max_iter=1000,random_state=seed)
    print "Logistic Regression"
    print crossValidation (lr, X, y)


    print "====================================="
    lda = LinearDiscriminantAnalysis()
    print "Linear Discriminant Analysis"
    print crossValidation (lda, X, y)




    est_gp = SymbolicRegressor(population_size=1000,
                           generations=100, stopping_criteria=0.01,
                           p_crossover=0.5, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)


    print "Symbolic Regressor"

    for pcross in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8]:
        for psubtreemutat in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8]:
            for phoistmutat in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8]:

                if ( pcross + psubtreemutat + phoistmutat ) <= 0.95:

                    est_gp = SymbolicRegressor(population_size=1000,
                                           generations=100, stopping_criteria=0.01,
                                           p_crossover=pcross, p_subtree_mutation=psubtreemutat,
                                           p_hoist_mutation=phoistmutat, p_point_mutation=0.1,
                                           max_samples=0.9, verbose=1,
                                           parsimony_coefficient=0.01, random_state=0)

                    #print "Symbolic Regressor"
                    score = crossValidation ( est_gp, X, y )


                    print "p_crossover: {} p_subtree_mutation: {} p_hoist_mutation: {} score: {}".format ( pcross, psubtreemutat, phoistmutat, score )

        (self, max_terms=None, max_degree=None, allow_missing=False,
                 penalty=None, endspan_alpha=None, endspan=None,
                 minspan_alpha=None, minspan=None,
                 thresh=None, zero_tol=None, min_search_points=None,
                 check_every=None,
                 allow_linear=None, use_fast=None, fast_K=None,
                 fast_h=None, smooth=None, enable_pruning=True,
                 feature_importance_type=None, verbose=0):



    print "====================================="

    print "MARS  degree 1"
    #MSE: 0.2459, GCV: 0.2459, RSQ: 0.0162, GRSQ: 0.0162

    model = Earth(max_terms=70, max_degree=1)
    model.fit(X,y)

    #Print the model
    #print(model.trace())
    print(model.summary())


    print "MARS  degree 3"

    model = Earth(max_terms=50, max_degree=3)
    model.fit(X,y)

    #Print the model
    #print(model.trace())
    print(model.summary())


    print "MARS  degree 5"

    model = Earth(max_terms=20, max_degree=5)
    model.fit(X,y)

    #Print the model
    #print(model.trace())
    print(model.summary())
   
    """

    print "====================================="

    print "MARS  degree 1"
    model = Earth(max_terms=70, max_degree=1)
    print "Score: {}".format ( crossValidation ( model, X, y ) )

    print "MARS  degree 3"
    model = Earth(max_terms=50, max_degree=3)
    crossValidation ( model, X, y )
    print "Score: {}".format ( crossValidation ( model, X, y ) )
