import data_extraction as de
import models as m
import pandas as pd

def train(df_train):
    df_splits = de.produce_splits(df_train)
    models = []
    train_method = m.train_k_nearest_neighbors # same model for each split
    for df in df_splits:
        models.append(train_method(df), 0) # registered
        models.append(train_method(df), 1) # casual
    return models
    

def predict(models, df_test):
    df_splits = de.produce_splits(df_test)
    
    if len(models)!= 2*len(df_splits):
        raise RuntimeError('Different number of models and splits')
        
    out_df = pd.DataFrame()
    for i in xrange(len(df_splits)):
        
        x = extract_knn_features(df_splits[i],0)
        r_count = models[2*i].predict(x)
        
        x = extract_knn_features(df_splits[i],1)
        c_count = models[2*i+1].predict(x)
        
        df_splits[i]['registered'] = r_count
        df_splits[i]['casual'] = c_count
        
        out_df = pd.concat([out_df, df_splits[i]])
    out_df = out_df.sort('index')
    return out_df
    
def run():
    df_train, df_test = de.read_data()
    models = train(df_train)
    out_df = predict(models, df_test)