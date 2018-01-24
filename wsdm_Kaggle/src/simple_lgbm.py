from __future__ import division

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')


#https://www.kaggle.com/kamilkk/simple-fast-lgbm-0-6685/code


import numpy as np
import pandas as pd
import lightgbm as lgb

import os, sys, getopt, cPickle, csv, sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/models/"



def getDataB ():
    data_path = 'input/'
    train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
                                                    'source_system_tab' : 'category',
                                                      'source_screen_name' : 'category',
                                                      'source_type' : 'category',
                                                      'target' : np.uint8,
                                                      'song_id' : 'category'})
    test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
                                                    'source_system_tab' : 'category',
                                                    'source_screen_name' : 'category',
                                                    'source_type' : 'category',
                                                    'song_id' : 'category'})
    songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category',
                                                      'language' : 'category',
                                                      'artist_name' : 'category',
                                                      'composer' : 'category',
                                                      'lyricist' : 'category',
                                                      'song_id' : 'category'})
    members = pd.read_csv(data_path + 'members.csv',dtype={'city' : 'category',
                                                          'bd' : np.uint8,
                                                          'gender' : 'category',
                                                          'registered_via' : 'category'})
    songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')


    return train, test, songs, members, songs_extra


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan



def getPrepDataB (train, test, songs, members, songs_extra ):

    print('Data training preprocessing...')
    song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
    song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
    train = train.merge(songs[song_cols], on='song_id', how='left')
    test = test.merge(songs[song_cols], on='song_id', how='left')

    members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
    members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
    members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

    members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
    members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
    members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
    members = members.drop(['registration_init_time'], axis=1)

    songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
    songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)

    train = train.merge(members, on='msno', how='left')
    test = test.merge(members, on='msno', how='left')

    train = train.merge(songs_extra, on = 'song_id', how = 'left')
    test = test.merge(songs_extra, on = 'song_id', how = 'left')

    import gc
    del members, songs; gc.collect();

    for col in train.columns:
        if train[col].dtype == object:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')


    X = train.drop(['target'], axis=1)
    y = train['target'].values

    return X, y, test




"""
def AUC(y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    return  auc 
"""


def AUC(y, pred):
    auc = roc_auc_score(y, pred)
    return  auc 




"""
def lgbmodel (X, y, file_name):
    d_train = lgb.Dataset(X, y)
    watchlist = [d_train]

    #Those parameters are almost out of hat, so feel free to play with them. I can tell
    #you, that if you do it right, you will get better results for sure ;)
    print('Training LGBM model...')
    params = {}
    params['learning_rate'] = 0.2
    params['application'] = 'binary'
    params['max_depth'] = 8
    params['num_leaves'] = 2**8
    params['verbosity'] = 0
    params['metric'] = 'auc'


    model = lgb.train(params, train_set=d_train, num_boost_round=50, valid_sets=watchlist, \
    verbose_eval=5)
"""


def lgbmodel ( X, y, metric = 'auc' ):
    d_train = lgb.Dataset(X, y)
    watchlist = [d_train]

    print('Training LGBM model...')
    params = {}
    params['learning_rate'] = 0.1
    params['application'] = 'binary'
    params['max_depth'] = 3
    params['num_leaves'] = 2**2
    params['verbosity'] = 0
    params['metric'] = metric


    model = lgb.train(params, train_set=d_train, num_boost_round=5000, valid_sets=watchlist, \
    verbose_eval=5)

    return model

#'xentlambda', 'binary_error'

def powerset(s):
    output = []
    x = len(s)
    for i in range(1 << x):
        tmp = [s[j] for j in range(x) if (i & (1 << j))]
        output.append (tmp)
    return output



def crossValidationlgb ( X, y, nfolds = 5, params = {}, num_boost_round=50 ):
    nrow, ncol = X.shape
    meanSize = nrow // nfolds

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
        d_train = lgb.Dataset(Xcur, ycur)
        watchlist = [d_train]

        model = lgb.train(params, train_set=d_train, num_boost_round=num_boost_round, valid_sets=watchlist)
        #get test
        if nextind < len(dataindex):

            tstart, tend = dataindex[nextind]
            Xtcur, ytcur = X.iloc[tstart:tend, :], y[tstart:tend]
            ypred = model.predict(Xtcur)
            score = AUC(ytcur, ypred)

        meanlist.append ( score )
    return sum (meanlist) / len (meanlist)



def featureSelectionlgb(X, y):
    bestfeatures = X.columns

    vscore = 0

    result = {}

    for currentlist in powerset( bestfeatures ):
        if len (currentlist) >= 5:

            out = getParametersOflgbmodel (X[currentlist], y, currentlist)

            score = out["score"]

            paramResult = out["result"]

            if vscore < score:
                vscore = score

                result["attributes"] = currentlist
                result["score"] = vscore
                result["param"] = paramResult

    return result



def getParametersOflgbmodel (X, y, currentlist):
    print "Hyperparameter search"
    result = ""

    vscore = 0

    for nlearning_rate in range (1, 10):
        nnlearning_rate = nlearning_rate / 10
        for nmax_depth in range (2, 15):
            for nnum_leaves in range (2, 15):
                for nnum_boost_round in range (1000, 10000, 50):
                    params = {}
                    params['learning_rate'] = nnlearning_rate
                    params['application'] = 'binary'
                    params['max_depth'] = nmax_depth
                    params['num_leaves'] = 2**nnum_leaves
                    params['verbosity'] = 0
                    params['metric'] = 'auc'


                    score = crossValidationlgb ( X, y, params = params, num_boost_round=nnum_boost_round )

                    print "score: {} \n learning_rate: {} \n max_depth: {} \n num_leaves: 2**{}  \n num_boost_round: {} ".format ( score, nnlearning_rate, nmax_depth, nnum_leaves, nnum_boost_round )
                    print "list of attributes"
                    print currentlist

                    if vscore < score:
                        vscore = score
                        result = "learning_rate: {} \n max_depth: {} \n num_leaves: 2**{}  \n num_boost_round: {} ".format ( nnlearning_rate, nmax_depth, nnum_leaves, nnum_boost_round )

    print "Best Parameters of the model\n"
    print result
    return {"score": vscore, "result": result }



def SingleFeatureSelectionlgb(X, y):
    bestfeatures = X.columns

    vscore = 0

    result = {}

    out = getParametersOflgbmodel (X, y, bestfeatures)

    score = out["score"]

    paramResult = out["result"]

    if vscore < score:
        vscore = score

        result["attributes"] = currentlist
        result["score"] = vscore
        result["param"] = paramResult

    return result





def writeResultToFile(test, file_name):
    ids = test['id'].values
    X_test = test.drop(['id'], axis=1)


    print('Making predictions and saving them...')

    filename = dir_path + file_name
    ispresent =  os.path.exists(filename)

    p_test = None

    if ispresent:

        model = lgb.Booster(model_file=filename)
        # can only predict with the best iteration (or the saving iteration)
        p_test = model.predict(X_test)

    subm = pd.DataFrame()
    subm['id'] = ids
    subm['target'] = p_test
    subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
    print('Done!')



#use cross validation
#https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/simple_example.py

import collections


if __name__ == "__main__":
    print('Loading data...')
    train, test, songs, members, songs_extra = getDataB ()

    X, y, ntest = getPrepDataB (train, test, songs, members, songs_extra )


    print collections.Counter( y )
    '''
    print X.head()
    print "============================================\n"
    print X.iloc[1:3, :]
    print "============================================\n"
    print type (y)
    print "============================================\n"
    print y[1:3]

    file_name = "lgbm.pkl"
    lgbmodel (X, y, file_name)

    writeResultToFile(ntest, file_name)

    getParametersOflgbmodel (X, y)

    params = {}
    params['learning_rate'] = 0.2
    params['application'] = 'binary'
    params['max_depth'] = 8
    params['num_leaves'] = 2**8
    params['verbosity'] = 0
    params['metric'] = 'auc'

    print crossValidationlgb (X, y, params = params)


    print "Best set of features"
    #featureSelectionlgb(X, y)
    '''

    #print "Best set of features"
    #print featureSelectionlgb(X, y)

    #print "full set of features"
    #print SingleFeatureSelectionlgb(X, y)

    params = {}

    params['learning_rate'] = 0.1
    params['application'] = 'binary'
    params['max_depth'] = 2
    params['num_leaves'] = 2**2
    params['verbosity'] = 0
    params['metric'] = 'auc'

    print crossValidationlgb ( X, y, nfolds = 5, params = params , num_boost_round=5000 )


