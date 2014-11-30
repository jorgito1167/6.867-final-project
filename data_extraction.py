import pandas as pd
import datetime
import pylab as plt
from sklearn import linear_model

def read_clean():
    '''
    Extracts the data and does some basic rearranging
    '''
    
    df_train = pd.read_csv('../train.csv', header=0)
    df_test = pd.read_csv('../test.csv', header=0)

    df_train['time'] = df_train['datetime'].apply(convert_date)
    df_test['time'] = df_test['datetime'].apply(convert_date)
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    
    
    x = df_train[features]
    y = df_train['count']
    
    model = linear_model.ElasticNetCV(n_jobs = 3)
    model.fit(x,y)
    
    x_test = df_test[features]
    df_test['count'] = model.predict(x_test)
    df_test['count'][ df_test['count']<0 ] = 0
    output = df_test[['datetime', 'count']]
    output.to_csv('out.csv', index = False)


def produce_subsets(x, season, holiday, workingday, weather):
    '''
    Produces subsets of the data that satisfy the following categorical
    data. Arrays can be input for multiple types of the same category.
    '''
    return x[(x['season'].isin(season)) & (x['holiday'].isin(holiday)) &
        (x['workingday'].isin(workingday)) & (x['weather'].isin(weather))]
    
def convert_date(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    time = d.hour + d.minute/60.0 + d.second/3600.0
    return time
    
if __name__ == '__main__':
    visualize()
    #run()
  
