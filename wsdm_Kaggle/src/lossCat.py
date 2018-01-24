from __future__ import division
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import datetime as dt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import f1_score

from catboost import CatBoostClassifier

from catboost import Pool, CatBoostClassifier
import numpy as np
import math


from pyearth import Earth

from sklearn import linear_model

from FileManager import FileManager

from simple_lgbm import lgbmodel

from simple_lgbm import getDataB, getPrepDataB

from problogic import getPrepDataC, probpredict

le = LabelEncoder()


def auc(x, y):
    return abs( np.trapz(y, x) )


class AUClossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)

        approx = approxes[0]

        weight_sum = 1.0

        error_sum = auc( target, approx )

        return error_sum, weight_sum



class HingelossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in xrange(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum +=  max ( 0, ( 1 - (target[i] * approx[i]) ) )

        return error_sum, weight_sum




class SquaredlossMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in xrange(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum +=  ( 1 - (target[i] * approx[i]) )**2 

        return error_sum, weight_sum




class LogisticMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)
        
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in xrange(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum +=  math.log( 1 + math.exp (-1 * target[i] * approx[i]) )

        error_sum = error_sum / math.log(2)
        return error_sum, weight_sum





def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


class GiniMetric(object):
    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.

        # weight parameter can be None.
        # Returns pair (error, weights sum)

        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 1.0

        error_sum = gini_normalized(target, approx)

        return error_sum, weight_sum





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




def crossEnsembleValidation ( X, y, nfolds = 5, loss_function='Logloss' ):

    #model = CatBoostClassifier ( iterations=400, loss_function='AUC' )

    model = CatBoostClassifier( loss_function=loss_function )

    nrow, ncol = X.shape
    meanSize = nrow // nfolds

    y = pd.to_numeric(y).tolist()

    score = 0.0

    meanlist = []

    dataindex = []

    for ind in range (0, nfolds):
        start, end = ind * meanSize, (ind+1) * meanSize
        dataindex.append ( (start, end) )


    for ind, curind in enumerate (dataindex):
        start, end = curind
        nextind = ind + 1
    
        Xcur, ycur = X[start:end], y[start:end]

        model.fit(Xcur, ycur)

        #get test
        if nextind < len(dataindex):

            tstart, tend = dataindex[nextind]
            Xtcur, ytcur = X[tstart:tend], y[tstart:tend]

            ypred = model.predict(Xtcur)


            ypred = ypred.reshape( (1, len(ypred) ) )



            ypred = ypred.T


            #score = AUC( ytcur, ypred )

            score = f1_score( ytcur, ypred, average='macro')  

        meanlist.append ( score )
    return sum (meanlist) / len (meanlist)



def crossEnsembleValidationCustom ( X, y, nfolds = 5, eval_metric=None ):

    model = CatBoostClassifier( eval_metric=eval_metric )

    nrow, ncol = X.shape
    meanSize = nrow // nfolds

    y = pd.to_numeric(y).tolist()

    score = 0.0

    meanlist = []

    dataindex = []

    for ind in range (0, nfolds):
        start, end = ind * meanSize, (ind+1) * meanSize
        dataindex.append ( (start, end) )


    for ind, curind in enumerate (dataindex):
        start, end = curind
        nextind = ind + 1
    
        Xcur, ycur = X[start:end], y[start:end]

        model.fit(Xcur, ycur)

        #get test
        if nextind < len(dataindex):

            tstart, tend = dataindex[nextind]
            Xtcur, ytcur = X[tstart:tend], y[tstart:tend]

            ypred = model.predict(Xtcur)


            ypred = ypred.reshape( (1, len(ypred) ) )



            ypred = ypred.T


            #score = AUC( ytcur, ypred )

            score = f1_score( ytcur, ypred, average='macro')  

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


def getDataA():

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


    #le = LabelEncoder()
    cat = ['msno', 'song_id', 'source_system_tab', 'source_screen_name',
           'source_type','gender','genre_ids','artist_name','composer',
           'lyricist']


    label(df_train,cat)
    label(df_test,cat)



    lstSet = set( ['target','registration_init_time', 'expiration_date'] )

    train_cols = set (df_train.columns) - lstSet
    train_cols = list (train_cols )

    X = df_train[train_cols]
    y = df_train['target'].apply(pd.to_numeric, errors='coerce')

    lstSet = set( ['id','registration_init_time', 'expiration_date'] )

    test_cols = set (df_test.columns) - lstSet
    test_cols = list ( test_cols )
    x_test = df_test[test_cols]

    #return X, y, x_test, df_test

    return X, y, x_test



def buildlevel1 ( data ):

    seed = 23
    modellst = []

    filename = "level1Ensemble.pkl"

    fObject = FileManager(filename)

    if not fObject.isExist( ):

        X, y = data['type1']['x'], data['type1']['y']

        ###add catboost here


        model = CatBoostClassifier( loss_function='Logloss' )
        model.fit(X, y)
        modellst.append ( model )


        model = CatBoostClassifier( loss_function='CrossEntropy' )
        model.fit(X, y)
        modellst.append ( model )


        model = CatBoostClassifier( eval_metric=HingelossMetric() )
        model.fit(X, y)
        modellst.append ( model )


        model = CatBoostClassifier( eval_metric=SquaredlossMetric() )
        model.fit(X, y)
        modellst.append ( model )


        model = CatBoostClassifier( eval_metric=LogisticMetric() )
        model.fit(X, y)
        modellst.append ( model )


        model = CatBoostClassifier( eval_metric=AUClossMetric() )
        model.fit(X, y)
        modellst.append ( model )


        model = CatBoostClassifier( eval_metric=GiniMetric() )
        model.fit(X, y)
        modellst.append ( model )



        X, y = data['type2']['x'], data['type2']['y']

        aucModel = lgbmodel ( X, y, metric = 'auc' )
        modellst.append ( aucModel )

        lambdaModel = lgbmodel ( X, y, metric = 'xentlambda' )
        modellst.append ( lambdaModel )

        binaryerrModel = lgbmodel ( X, y, metric = 'binary_error' )
        modellst.append ( binaryerrModel )

        #save model to disk
        fObject.save ( modellst ) 



"""
def getDataLevel1Output (data, testing=False):

    filename = "level1Ensemble.pkl"

    fObject = FileManager(filename)

    modellst  =  fObject.load( ) #load from disk

    X, y = data['type1']['x'], data['type1']['y']

    #modify the index of the modellst to allow MARS
    arr = []
    for model in modellst[:5]:
        y_pred = model.predict( X )
        arr.append( y_pred )


    X, y = data['type2']['x'], data['type2']['y']

    for model in modellst[5:]:
        y_pred = model.predict( X )
        arr.append( y_pred )



    np_arr = np.array(arr)

    if not testing:
        return np_arr.T, y 

    return np_arr.T
"""



def getDataLevel1Output (data, testing=False):

    filename = "level1Ensemble.pkl"

    fObject = FileManager(filename)

    modellst  =  fObject.load( ) #load from disk


    if testing:

        X = data['type1']['x']

        #modify the index of the modellst to allow MARS
        arr = []
        for model in modellst[:7]:
            y_pred = model.predict( X )
            arr.append( y_pred )


        X = data['type2']['x']

        for model in modellst[7:]:
            y_pred = model.predict( X )
            arr.append( y_pred )



        np_arr = np.array(arr)
        return np_arr.T


    X, y = data['type1']['x'], data['type1']['y']

    #modify the index of the modellst to allow MARS
    arr = []
    for model in modellst[:7]:
        y_pred = model.predict( X )
        arr.append( y_pred )


    X, y = data['type2']['x'], data['type2']['y']

    for model in modellst[7:]:
        y_pred = model.predict( X )
        arr.append( y_pred )



    np_arr = np.array(arr)

    return np_arr.T, y 

    



def predict ( xval, yval = np.array([]), df_test = None, threshold = 0.5):
    #This takes the output from level 1 in the training phase
    #exclude the label and take output from level 1

    """
    modellst = [ CatBoostClassifier( loss_function='Logloss' ), CatBoostClassifier( loss_function='CrossEntropy' ), CatBoostClassifier( eval_metric=HingelossMetric() ), CatBoostClassifier( eval_metric=SquaredlossMetric() ), CatBoostClassifier( eval_metric=LogisticMetric() ), CatBoostClassifier( eval_metric=AUClossMetric() ), CatBoostClassifier( eval_metric=GiniMetric() ) ]
    """
    modellst = [ CatBoostClassifier( eval_metric=AUClossMetric() ) ]
    filename = "level2Ensemble.pkl"

    fObject = FileManager(filename)

    if yval.size > 0:
        #training stage
        #if model does not exist in disk, create model and save to disk after
        nModlist = []
        for model in modellst:
            model.fit( xval, yval )
            nModlist.append ( model )

        #save model to disk
        fObject.save ( nModlist ) 

    else:
        #testing stage 
        arr = []

        lmodellst = fObject.load( ) #load from disk
        
        for model in lmodellst:
            y_pred = model.predict(xval)
            arr.append( y_pred )

        np_arr = np.array(arr)

        #majority voting

        """
        np_sum = np.sum(np_arr, axis=0) #sum by column

        #threshold and normalize to (0, 1)
        tOutp =  np_sum / len (np_sum) 
        tnp_arr = tOutp.T
        """


        np_sum = np.average(np_arr, axis=0) #average by column

        tnp_arr = np_sum.T



        data_path = 'output/'
        submit = pd.DataFrame({'id':df_test['id'],'target':tnp_arr})
        submit['target'] = submit['target'] >= threshold 
        submit['target'] = submit['target'].astype(int)
        submit.to_csv(data_path + 'final-kaggle2.csv',index=False)


def training ( data ):
    buildlevel1 ( data )
    xval, yval = getDataLevel1Output ( data )
    predict ( xval, yval=yval )


def testing ( data, xtest ):
    buildlevel1 ( data )
    xval = getDataLevel1Output (data, testing=True)

    predict ( xval, df_test = xtest )



if __name__ == "__main__":

    Xa, ya, xtesta = getDataA()

    train, test, songs, members, songs_extra = getDataB ()
    Xb, yb, xtestb = getPrepDataB (train, test, songs, members, songs_extra )

    traindata = { 'type1': {'x': Xa, 'y': ya}, 'type2': {'x': Xb, 'y': yb} }

    testdata = { 'type1': {'x': xtesta }, 'type2': {'x': xtestb } } 


    training ( traindata )

    testing ( testdata, xtestb )

