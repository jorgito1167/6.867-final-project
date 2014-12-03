import pandas as pd
import datetime
import pylab as plt
from sklearn import linear_model

def read_data():
    '''
    Extracts the data and does some basic rearranging
    '''
    
    df_train = pd.read_csv('../train.csv', header=0)
    df_test = pd.read_csv('../test.csv', header=0)
    
    df_train['index'] = df_train.index
    df_test['index'] = df_test.index
    
    # Create features (week, dow, 
    df_train = create_features(df_train)
    df_test = create_features(df_test)
    
    # Remove atemp
    df_train = df_train.drop('atemp',1)
    df_test = df_test.drop('atemp',1)
    
    # set weather 4 = 3
    df_train = change_weather(df_train)
    df_test = change_weather(df_test)

    return df_train, df_test 

def produce_subsets(x, season, holiday, workingday, weather):
    '''
    Produces subsets of the data that satisfy the following categorical
    data. Arrays can be input for multiple types of the same category.
    '''
    return x[(x['season'].isin(season)) & (x['holiday'].isin(holiday)) &
        (x['workingday'].isin(workingday)) & (x['weather'].isin(weather))]

def create_features(x):
    x['hour'] = x['datetime'].apply(create_hour)
    x['dow'] = x['datetime'].apply(create_dow)
    x['week'] = x['datetime'].apply(create_week)
    x['yearpart'] = x['datetime'].apply(create_year_part)
    return x

def change_weather(df):
    df[df['weather'] == 4]['weather'] = 3
    return df
    
def produce_splits(df):
    df_list = split_var('season', [df])
    df_list = split_var('holiday', df_list)
    df_list = split_var('workingday', df_list) 
    df_list = split_weather(df_list)  
    #split_weekday(df_list)
    return df_list

def split_var(var, df_list):        
    new_splits = []
    unique = df_list[0][var].unique()
    for df in df_list:
        for i in unique:
            new_splits.append(df[df[var]==i])
    return new_splits
    
def split_weather(df_list):
    new_splits = []
    for df in df_list:
        for i in xrange(1,4):
            if i ==3:
                new_splits.append(df[df['season'].isin([i,i+1])])
            else:
                new_splits.append(df[df['season']==i])
    df_list = new_splits
    return new_splits
    
def split_weekday(df_list):
    new_splits = []
    for df in df_list:
        new_splits.append(df[df['weekday'].isin([0,1,2,3,4])])
        new_splits.append(df[df['weekday'].isin([5,6])])
    df_list = new_splits
    
def create_hour(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return d.hour

def create_dow(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return d.weekday()

def create_week(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    week = ((d - datetime.datetime(d.year,1,1)).days / 7) + 1
    
    # Get distance from 30th week
    return abs(30 - week)

def create_year_part():
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    week = ((d - datetime.datetime(d.year,1,1)).days / 7) + 1
    
    # 0 if first half of year and 1 if second half of year
    return int(30 - week > 0)
    
#if __name__ == '__main__':
#    read_data()
#  
