import data_extraction as de
import models 
import pandas as pd

def train(df_train):
    df_splits = de.produce_splits(df_train)
    model_list = []
    train_method = models.train_k_nearest_neighbors # same model for each split
    for df in df_splits:
        model_list.append(train_method(df, 0)) # registered
        model_list.append(train_method(df, 1)) # casual
    return models
    

def predict(model_list, df_test):
    df_splits = de.produce_splits(df_test)
    
    if len(models)!= 2*len(df_splits):
        raise RuntimeError('Different number of models and splits')
        
    out_df = pd.DataFrame()
    for i in xrange(len(df_splits)):
        
        r_count = model_list[2*i].predict(x)
        c_count = model_list[2*i+1].predict(x)
        
        df_splits[i]['registered'] = r_count
        df_splits[i]['casual'] = c_count
        
        out_df = pd.concat([out_df, df_splits[i]])
    out_df = out_df.sort('index')
    return out_df
    
def run():
    df_train, df_test = de.read_data()
    models = train(df_train)
    out_df = predict(models, df_test)