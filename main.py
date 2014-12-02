import data_extraction as de
import models 
import pandas as pd
import numpy as np

def train(df_train):
    df_splits = de.produce_splits(df_train)
    model_list = []
    train_method = models.train_ridge_regression # same model for each split
    counter = 1
    for df in df_splits:
        model_list.append(train_method(df, 0)) # registered
        model_list.append(train_method(df, 1)) # casual
        print counter
        counter+=1
    return models
    

def predict(model_list, df_test):
    df_splits = de.produce_splits(df_test)
    
    if len(models)!= 2*len(df_splits):
        raise RuntimeError('Different number of models and splits')
        
    out_df = pd.DataFrame()
    for i in xrange(len(df_splits)):
        
        if df_splits[i].empty:
            continue
        r_count = model_list[2*i].predict(x)
        c_count = model_list[2*i+1].predict(x)
        
        df_splits[i]['registered'] = r_count
        df_splits[i]['casual'] = c_count
        
        out_df = pd.concat([out_df, df_splits[i]])
    out_df = out_df.sort('index')
    return out_df

def metric(estimator, x, y):
    predicted_y = estimator.predict(x) 
    print np.power(sum(np.power((np.log(predicted_y+1)- np.log(y+1)),2))/len(y), 0.5)
    return np.power(sum(np.power((np.log(predicted_y+1)- np.log(y+1)),2))/len(y), 0.5)
    
    
def run():
    df_train, df_test = de.read_data()
    model_list = train(df_train)
    out_df = predict(model_list, df_test)
