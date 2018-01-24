from __future__ import division
import pandas as pd
import numpy as np
from sklearn import metrics


import datetime as dt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score

import keras
import tensorflow as tf
from keras.models import Sequential
# For custom metrics
import keras.backend as K
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

from simple_lgbm import getDataB, getPrepDataB

from keras.layers import LSTM
from keras.layers import TimeDistributed


from keras.layers import Conv1D, MaxPooling1D

from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Dropout, Bidirectional, LSTM

from keras.layers.wrappers import TimeDistributed

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import GRU

from keras.layers.pooling import GlobalAveragePooling1D

from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D



from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import PReLU


from keras.models import model_from_json








# fix random seed for reproducibility
seed = 7
np.random.seed(seed)



le = LabelEncoder()

# Convolution
kernel_size = 5
filters = 64
pool_size = 4



def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score



# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41108
def jacek_auc(y_true, y_pred):
   score, up_opt = tf.metrics.auc(y_true, y_pred)
   #score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)    
   K.get_session().run(tf.local_variables_initializer())
   with tf.control_dependencies([up_opt]):
       score = tf.identity(score)
   return score



# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41015
# AUC for a binary classifier
def discussion41015_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)



#---------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N



#----------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P





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



def crossValidationMLP (model, X, y, cb, nfolds = 5 ):
    nrow, ncol = X.shape
    meanSize = nrow // nfolds

    #y = y.to_frame()

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
    
        Xcur, ycur = X[start:end, :], y[start:end]


        #model.fit(Xcur, ycur, nb_epoch=10, batch_size=16, validation_split=0.1)
        model.fit(Xcur, ycur, epochs=100, validation_split=0.1, callbacks=cb)


        #get test
        if nextind < len(dataindex):

            tstart, tend = dataindex[nextind]
            Xtcur, ytcur = X[tstart:tend, :], y[tstart:tend, :]

            ypred = model.predict(Xtcur)


            ypred = ypred.reshape( (1, len(ypred) ) )

            #ytcur = ytcur.reshape( (1, len(ytcur) ) ) 

            ypred = ypred.T

            #print ytcur.shape, ypred.shape
            #print type (ytcur), type( ypred ) 
            #nmlist = ytcur.tolist()


            #score = AUC( ytcur, ypred )

            score = f1_score( ytcur.astype(int), ypred.astype(int), average='macro')  

        meanlist.append ( score )
    return sum (meanlist) / len (meanlist)



def model_relu1():

    model = Sequential()

    model.add(Dense(512, activation='relu', input_dim=26))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[jacek_auc,discussion41015_auc])
    return model



def model_relu3():

    batch_size=32
    filter_length = 5
    nb_filter = 64
    pool_length = 4
    nb_epoch = 3

    inputDim = 26

    model = Sequential()
    model.add( Embedding(inputDim*inputDim, inputDim, dropout=0.2) ) #input vector dimension

    model.add(Convolution1D(nb_filter= nb_filter, filter_length= filter_length, border_mode='valid', activation='relu',subsample_length=1))

    model.add(MaxPooling1D(pool_length= pool_length))

    model.add(Bidirectional( LSTM(1024, return_sequences=True) ))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Bidirectional( LSTM(2048, return_sequences=True) ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Bidirectional( LSTM(512) ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[jacek_auc,discussion41015_auc])

    return model



def model_relu4():

    batch_size=32
    filter_length = 5
    nb_filter = 64
    pool_length = 4
    nb_epoch = 3

    inputDim = 26

    model = Sequential()
    model.add( Embedding(inputDim*inputDim*inputDim, inputDim, dropout=0.2) ) #input vector dimension

    model.add(Convolution1D(nb_filter= nb_filter, filter_length= filter_length, border_mode='valid', activation='relu',subsample_length=1))

    model.add(MaxPooling1D(pool_length= pool_length))

    model.add( LSTM(1024, return_sequences=True) )
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add( LSTM(2048, return_sequences=True) )
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add( LSTM(512) )
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[jacek_auc,discussion41015_auc])

    return model


def model_relu5():

    model = Sequential()

    inputDim = 26

    model.add( Embedding(inputDim*inputDim*inputDim, inputDim, dropout=0.2) ) #input vector dimension


    model.add( LSTM(1024, return_sequences=True) )
    model.add(LeakyReLU())
    model.add(BatchNormalization( ))
    model.add(Dropout(0.5))

    model.add( LSTM(2048, return_sequences=True) )
    model.add(Activation('relu'))
    model.add(BatchNormalization( ))
    model.add(Dropout(0.5))

    model.add( LSTM(512, return_sequences=True) )
    model.add(Activation('relu'))
    model.add(BatchNormalization( ))
    model.add(Dropout(0.5))

    model.add( LSTM(256) )
    model.add(Activation('relu'))
    model.add(BatchNormalization( ))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[jacek_auc,discussion41015_auc])

    return model



def model_relu6():

    batch_size=32
    filter_length = 5
    nb_filter = 64
    pool_length = 4
    nb_epoch = 3

    inputDim = 26

    model = Sequential()
    model.add( Embedding(inputDim*inputDim*inputDim, inputDim, dropout=0.2) ) #input vector dimension

    model.add(Convolution1D(nb_filter= nb_filter, filter_length= filter_length, border_mode='valid', activation='relu',subsample_length=1))

    model.add(MaxPooling1D(pool_length= pool_length))


    model.add( Bidirectional( LSTM(1024, return_sequences=True) ))
    model.add(LeakyReLU())
    model.add(BatchNormalization( ))
    model.add(Dropout(0.5))

    model.add( Bidirectional( LSTM(2048, return_sequences=True)) )
    model.add(Activation('relu'))
    model.add(BatchNormalization( ))
    model.add(Dropout(0.5))

    model.add( Bidirectional( LSTM(512, return_sequences=True)) )
    model.add(Activation('relu'))
    model.add(BatchNormalization( ))
    model.add(Dropout(0.5))

    model.add( Bidirectional( LSTM(256)) )
    model.add(Activation('relu'))
    model.add(BatchNormalization( ))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[jacek_auc,discussion41015_auc])

    return model


def model_relu2():

    model = Sequential()

    model.add(Dense(512, activation='relu', input_dim=26))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(1024))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[jacek_auc,discussion41015_auc])

    return model


#go here, it's easier to understand callbacks reading keras source code:
#   https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L838
#   https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1040

class GiniWithEarlyStopping(keras.callbacks.Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=1024):
        #print("self vars: ",vars(self))  #uncomment and discover some things =)
        
        # FROM EARLY STOP
        super(GiniWithEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size=predict_batch_size
    
    def on_batch_begin(self, batch, logs={}):
        if(self.verbose > 1):
            if(batch!=0):
                print("")
            print("Hi! on_batch_begin() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
    
    def on_batch_end(self, batch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_batch_end() , batch=",batch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
    
    def on_train_begin(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_begin() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
    
    def on_train_end(self, logs={}):
        if(self.verbose > 1):
            print("Hi! on_train_end() ,logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

        # FROM EARLY STOP
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ',self.stopped_epoch,': GiniEarlyStopping')
    
    def on_epoch_begin(self, epoch, logs={}):
        if(self.verbose > 1):
            print("Hi! on_epoch_begin() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)

    def on_epoch_end(self, epoch, logs={}):
        if(self.validation_data):
            y_hat_val=self.model.predict(self.validation_data[0],batch_size=self.predict_batch_size)
            
        if(self.verbose > 1):
            print("Hi! on_epoch_end() , epoch=",epoch,",logs:",logs)
            #print("self vars: ",vars(self))  #uncomment and discover some things =)
        
        #i didn't found train data to check gini on train set (@TODO HERE)
        # from source code of Keras: https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L1127
        # for cbk in callbacks:
        #     cbk.validation_data = val_ins
        # Probably we will need to change keras... 
        # 
        
            print("    GINI Callback:")
            if(self.validation_data):
                print('        validation_data.inputs       : ',np.shape(self.validation_data[0]))
                print('        validation_data.targets      : ',np.shape(self.validation_data[1]))
                print("        roc_auc_score(y_real,y_hat)  : ",roc_auc_score(self.validation_data[1], y_hat_val ))
                print("        gini_normalized(y_real,y_hat): ",gini_normalized(self.validation_data[1], y_hat_val))
                print("        roc_auc_scores*2-1           : ",roc_auc_score(self.validation_data[1], y_hat_val)*2-1)
        
            print('    Logs (others metrics):',logs)
        # FROM EARLY STOP
        if(self.validation_data):
            if (self.verbose == 1):
                print("\n GINI Callback:",gini_normalized(self.validation_data[1], y_hat_val))
            current = gini_normalized(self.validation_data[1], y_hat_val)
            
            # we can include an "gambiarra" (very usefull brazilian portuguese word)
            # to logs (scores) and use others callbacks too....
            # logs['gini_val']=current
            
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True











class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size=predict_batch_size
        self.include_on_batch=include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if(self.include_on_batch):
            logs['roc_auc_val']=float('-inf')
            if(self.validation_data):
                logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                                  self.model.predict(self.validation_data[0],
                                                                     batch_size=self.predict_batch_size))

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val']=float('-inf')
        if(self.validation_data):
            logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                              self.model.predict(self.validation_data[0],
                                                                 batch_size=self.predict_batch_size))


Xv, yv, test = getDataA()
trainW, testW, songs, members, songs_extra = getDataB ()
Xb, yb, xtestb = getPrepDataB (trainW, testW, songs, members, songs_extra )

y_col = yv.values






# batch_size=500 ~= 2 batchs
estimator = KerasClassifier(build_fn=model_relu6, nb_epoch=5, batch_size=500, verbose=1)

cb = [
    RocAucMetricCallback(), # include it before EarlyStopping!
    EarlyStopping(monitor='roc_auc_val',patience=1, verbose=2) 
]



estimator.fit( Xv.values, y_col ,epochs=100,validation_split=.2,callbacks=cb )

y_pred = estimator.predict(test.values)

data_path = 'output/'
submit = pd.DataFrame({'id':xtestb['id'],'target':y_pred.tolist() })
#submit['target'] = submit['target'].astype(int)

submit.to_csv(data_path + 'final-testmmVV5W2.csv',index=False)


#################################################################






# batch_size=500 ~= 2 batchs
estimator = KerasClassifier(build_fn=model_relu5, nb_epoch=5, batch_size=500, verbose=1)

cb = [
    RocAucMetricCallback(), # include it before EarlyStopping!
    EarlyStopping(monitor='roc_auc_val',patience=1, verbose=2) 
]



estimator.fit( Xv.values, y_col ,epochs=100,validation_split=.2,callbacks=cb )

y_pred = estimator.predict(test.values)

data_path = 'output/'
submit = pd.DataFrame({'id':xtestb['id'],'target':y_pred.tolist() })
#submit['target'] = submit['target'].astype(int)

submit.to_csv(data_path + 'final-testmmVV52.csv',index=False)

#################################################################

# batch_size=500 ~= 2 batchs
estimator = KerasClassifier(build_fn=model_relu5, nb_epoch=5, batch_size=500, verbose=1)



cb = [
    # verbose =2 make many prints (nice to learn keras callback)
    GiniWithEarlyStopping(patience=1, verbose=2) 
]


estimator.fit( Xv.values, y_col ,epochs=100,validation_split=.2,callbacks=cb )

y_pred = estimator.predict(test.values)
print "y_pred   {}".format (y_pred.shape)
data_path = 'output/'
submit = pd.DataFrame({'id':xtestb['id'],'target':y_pred.tolist() })
#submit['target'] = submit['target'].astype(int)
submit.to_csv(data_path + 'final-testmmVV51.csv',index=False)

#################################################################

# batch_size=500 ~= 2 batchs
estimator = KerasClassifier(build_fn=model_relu6, nb_epoch=5, batch_size=500, verbose=1)



cb = [
    # verbose =2 make many prints (nice to learn keras callback)
    GiniWithEarlyStopping(patience=1, verbose=2) 
]


estimator.fit( Xv.values, y_col ,epochs=100,validation_split=.2,callbacks=cb )

y_pred = estimator.predict(test.values)
print "y_pred   {}".format (y_pred.shape)
data_path = 'output/'
submit = pd.DataFrame({'id':xtestb['id'],'target':y_pred.tolist() })
#submit['target'] = submit['target'].astype(int)
submit.to_csv(data_path + 'final-testmmVV5W1.csv',index=False)




