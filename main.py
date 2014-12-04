import data_extraction as de
import models 
import pandas as pd
import numpy as np
import config
import timeit
import datetime

def train(df_train):
    print 'Start Training'
    start = timeit.default_timer()
    log_file = open('log_file4.csv' , 'w')#+ datetime.datetime.now().isoformat(), 'w')
    log_file.write(config.split_variables())

    df_splits = de.process_df(df_train) # splits, normalizes, binarizes, and expands
    model_list = []
    train_method = models.train_random_forest_regressor # same model for each split
    counter = 1
    for df in df_splits:
        if df.empty:
            model_list.extend([None,None])
            log_file.write(str(counter) + ',0,\n')
        else:
            mod1,str1 = train_method(df, 0)
            model_list.append(mod1) # registered
            mod2,str2 = train_method(df, 1)
            model_list.append(mod2) # casual
            log_file.write(str(counter) + ', registered,' + config.get_vars(df) + str1)
            log_file.write(str(counter) + ', casual,'+ config.get_vars(df) + str2)
        counter += 1
    log_file.close()
    print "Total time to train: " + str(timeit.default_timer()-start)
    return model_list
    

def predict(model_list, df_test, df_train):
    df_splits = de.process_df(df_test) # splits, normalizes, binarizes, and expands
    train_splits = de.process_df(df_train)
    
    if len(model_list)!= 2*len(df_splits):
        raise RuntimeError('Different number of models and splits')
    
    out_df = pd.DataFrame()
    for i in xrange(len(df_splits)):
        if df_splits[i].empty:
            continue
            
        check_consistency(df_splits[i], train_splits[i]) #check to see that the training split and the testing split have the same charasteristics
        
        x = df_splits[i].drop(config.non_features_counts, 1).values
        
        if model_list[2*i] != None:
            r_count = model_list[2*i].predict(x)
        else:
            print 'No Model for split: ' + str(i)
            print len(df_splits[i]['index'])
            r_count = np.log(155*np.ones((len(df_splits[i]['index']),)))
        
        if model_list[2*i+1] != None:
            c_count = model_list[2*i+1].predict(x)
            
        else:
            print 'No Model for split: ' + str(i)
            print len(df_splits[i]['index'])
            c_count = np.log(36*np.ones((len(df_splits[i]['index']),)))
        
        if config.use_log:
            r_count[r_count < 0] = 0
            c_count[c_count < 0] = 0
        else:
            df_splits[i]['registered'] = np.power(np.e,r_count)
            df_splits[i]['casual'] = np.power(np.e,c_count)
        
        out_df = pd.concat([out_df, df_splits[i]])
    out_df = out_df.sort('index')
    out_df['count'] = out_df['registered'] + out_df['casual']
    output = out_df[['datetime', 'count']]
    output.to_csv('out.csv', index = False)
    return out_df

def check_consistency(df1, df2):
    check_holiday = (df1['holiday'].unique() == df2['holiday'].unique()) and (len(df1['holiday'].unique()) ==1)
    check_workingday = (df1['workingday'].unique() == df2['workingday'].unique()) and (len(df1['workingday'].unique()) ==1)
    check_segment = (df1['segment'].unique() == df2['segment'].unique()) and (len(df1['segment'].unique()) ==1)
    print check_holiday
    print check_workingday
    print (df1['workingday'].unique() == df2['workingday'].unique())
    print df1['workingday'].unique()
    print df2['workingday'].unique()
    print check_segment
    if not(check_holiday and check_workingday and check_segment):
        raise RuntimeError('DANG! Jeff Chan NO!')
        
def metric(estimator, x, y):
    predicted_y = estimator.predict(x) 
    if config.use_log:
        y = np.power(np.e, y.copy())
        predicted_y = np.power(np.e, predicted_y)
    else:
        predicted_y[predicted_y<0] = 0
    return np.power(sum(np.power((np.log(predicted_y+1)- np.log(y+1)),2))/len(y), 0.5)
    
def run():
    df_train, df_test = de.read_data()
    model_list = train(df_train)
    out_df = predict(model_list, df_test)

if __name__ == '__main__':
    df_train, df_test = de.read_data()
    model_list = train(df_train)
    out_df = predict(model_list, df_test, df_train)
