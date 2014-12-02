import data_extraction as de
import models 
import pandas as pd
import numpy as np
import config
import timeit

def train(df_train):
    print 'Start Training'
    start = timeit.default_timer()
    df_splits = de.produce_splits(df_train)
    model_list = []
    train_method = models.train_ridge_regression # same model for each split
    counter = 1
    for df in df_splits:
        model_list.append(train_method(df, 0)) # registered
        model_list.append(train_method(df, 1)) # casual
    print "Total time to train: " + str(timeit.default_timer()-start)
    return model_list
    

def predict(model_list, df_test):
    df_splits = de.produce_splits(df_test)
    
    if len(model_list)!= 2*len(df_splits):
        raise RuntimeError('Different number of models and splits')
    
    out_df = pd.DataFrame()
    for i in xrange(len(df_splits)):
        if df_splits[i].empty:
            continue
       
        x = df_splits[i].drop(config.non_features_counts, 1).values
        if model_list[2*i] != None:
            print model_list[2*i]
            r_count = model_list[2*i].predict(x)
            r_count[r_count < 0] = 0
        else:
            r_count = 155*np.ones((len(df_splits[i]['index']),))
        
        if model_list[2*i+1] != None:
            c_count = model_list[2*i+1].predict(x)
            c_count[c_count < 0] = 0
        else:
            c_count = 36*np.ones((len(df_splits[i]['index']),))
        
        df_splits[i]['registered'] = r_count
        df_splits[i]['casual'] = c_count
        
        out_df = pd.concat([out_df, df_splits[i]])
    out_df = out_df.sort('index')
    out_df['count'] = out_df['registered'] + out_df['casual']
    output = out_df[['datetime', 'count']]
    output.to_csv('out.csv', index = False)
    return out_df

def metric(estimator, x, y):
    predicted_y = estimator.predict(x) 
    predicted_y[predicted_y<0] = 0
    return np.power(sum(np.power((np.log(predicted_y+1)- np.log(y+1)),2))/len(y), 0.5)
    
    
def run():
    df_train, df_test = de.read_data()
    model_list = train(df_train)
    out_df = predict(model_list, df_test)
