from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import f1_score



def crossValidation ( X, y, nfolds = 5, threshold = 0.5 ):
    nrow, ncol = X.shape
    meanSize = nrow // nfolds

    #y = y.to_frame()


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

        overall_prob, all3_cat_prob = getModelData ( Xcur )

        #get test
        if nextind < len(dataindex):

            tstart, tend = dataindex[nextind]
            Xtcur, ytcur = X[tstart:tend], y[tstart:tend]

            test_df = prediction (overall_prob, all3_cat_prob, Xtcur, threshold )

            ypred = test_df['ratio']


            score = f1_score( ytcur.astype(int), ypred.astype(int), average='macro')  

        meanlist.append ( score )
    return sum (meanlist) / len (meanlist)






def getModelData (train_df):


    temp_df = train_df.groupby(['source_system_tab', 'source_screen_name'])['target'].aggregate(['count', 'sum']).reset_index()

    temp_df['ratio'] = temp_df['sum']/temp_df['count']

    temp_df_pivot = temp_df.pivot('source_system_tab', 'source_screen_name', 'ratio')


    target_count_df = train_df.groupby(['source_system_tab', 'source_screen_name', 'source_type'])['target'].aggregate('count').reset_index()

    target_sum_df = train_df.groupby(['source_system_tab', 'source_screen_name', 'source_type'])['target'].aggregate('sum').reset_index()

    target_count_df['target_count'] = target_sum_df['target']

    target_count_df['ratio'] = target_count_df['target_count']/target_count_df['target']

    all3_cat_prob = target_count_df.copy()

    del target_count_df, target_sum_df

    all3_cat_prob.sort_values('target', ascending=False)


    all3_cat_prob['target_prob'] = all3_cat_prob['target']/all3_cat_prob['target'].aggregate('sum')

    all3_cat_prob['target_count_prob'] = all3_cat_prob['target_count']/all3_cat_prob['target_count'].aggregate('sum')

    overall_prob = all3_cat_prob['target_count'].aggregate('sum')/all3_cat_prob['target'].aggregate('sum')

    all3_cat_prob = all3_cat_prob.drop(['target'], axis=1)

    return overall_prob, all3_cat_prob



def preprocess ( train_df ):

    train_df['language'].fillna('median', inplace=True)
    train_df['source_system_tab'].fillna('median', inplace=True)
    train_df['genre_ids'].fillna('median', inplace=True)
    train_df['source_type'].fillna('median', inplace=True)
    train_df['source_screen_name'].fillna('median', inplace=True)
    return train_df



def prediction (overall_prob, all3_cat_prob, test_df, threshold ):

    test_df = pd.merge(test_df, all3_cat_prob, on =['source_system_tab', 'source_screen_name', 'source_type'], how='left')

    test_df['ratio'].fillna(overall_prob, inplace=True)

    test_df.isnull().aggregate('sum')


    test_df['ratio'] = test_df['ratio'] > threshold  

    test_df['ratio'] = test_df['ratio'].astype(int)     

    return test_df


def writeToData (test_df):

    data_path = 'output/'
    submit_df = pd.DataFrame()

    submit_df['id'] = test_df['id']
    submit_df['target'] = test_df['ratio']

    submit_df.to_csv(data_path + 'easy-submission.csv', index=False, float_format='%.5f')


def evaluate(X, y):

    vscore = 0

    result = ""

    for tindx in range(1, 20):

        thres = tindx / 20
        score = crossValidation (X, y, threshold = thres )

        if vscore < score:
            vscore = score
            result = "F score: {} threshold {}".format(vscore, thres)

    return result



def getPrepDataC():

    data_path = 'input/'

    train_df = pd.read_csv(data_path +'train.csv')

    test_df = pd.read_csv(data_path +'test.csv')

    return train_df, test_df



def probpredict (train_df, test_df, threshold = 0.5):

    overall_prob, all3_cat_prob = getModelData (train_df)

    submit_df = prediction (overall_prob, all3_cat_prob, test_df, threshold )

    return submit_df



if __name__ == "__main__":

    train_df, test_df = getPrepDataC()
    submit_df = probpredict (train_df, test_df)
    writeToData ( submit_df  )




