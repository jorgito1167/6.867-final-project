import pandas as pd
import datetime
import pylab as plt
from sklearn import linear_model, preprocessing

def read_data():
    '''
    Extracts the data and does some basic rearranging
    '''
    
    df_train = pd.read_csv('../train.csv', header=0)
    df_test = pd.read_csv('../test.csv', header=0)
    
    # Add index
    df_train['index'] = df_train.index
    df_test['index'] = df_test.index
    
    df_train = change_weather(df_train)
    df_test = change_weather(df_test)
    
    # Create features (week, dow, yearpart)
    df_train = create_features(df_train)
    df_test = create_features(df_test)

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
    x['segment'] = x['hour'].apply(create_segment) # segment of day
    x['sunday'] = x['datetime'].apply(create_sunday) # tells you if its sunday 
    x['week'] = x['datetime'].apply(create_week)
    x['yearpart'] = x['datetime'].apply(create_year_part) # tells you if its first half or second half of the year
    
    # Windspeed and Temperature binary features
    x['highwind'] = x['windspeed'].apply(lambda w: int(w > 40))
    x['hightemp'] = x['temp'].apply(lambda t: int(t > 30))
    return x

def process_df(df):
    df_splits = produce_splits(df) # split variables
    df_splits = normalize_features(df_splits) # normalize
    df_splits = binarize_features(df_splits)# binarize
    df_splits = expand_features(df_splits) # expand feature set
    return df_splits
    
def produce_splits(df):
    #df_list = split_var('season', [df])
    df_list = split_var('holiday', [df])
    df_list = split_var('workingday', df_list)
    df_list = split_var('segment', df_list)
    #df_list = split_weather(df_list)  
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
    
def change_weather(df):
    x = df[df['weather'] == 4]
    x['weather'] = 3
    return x

def create_hour(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return d.hour

def create_segment(hour):
    '''
    0 =  5AM - 10AM
    1 = 10AM - 4PM
    2 = 4PM - 5AM
    '''
    if hour >= 5 and hour < 10:
        return 0
    elif hour >= 10 and hour < 16:
        return 1
    else:
        return 2

def create_sunday(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return int(d.weekday() == 6)

def create_week(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    week = ((d - datetime.datetime(d.year,1,1)).days / 7) + 1
    
    # Get distance from 30th week
    return abs(30 - week)

def create_year_part(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    week = ((d - datetime.datetime(d.year,1,1)).days / 7) + 1
    
    # 0 if first half of year and 1 if second half of year
    return int(30 - week > 0)

def normalize_features(df_splits, normalized_features = [
                            'temp', 'humidity', 'windspeed', 'hour', 'week']):
    norm = preprocessing.StandardScaler(copy = False)
    for df in df_splits:
        norm.fit_transform(df.loc[:,normalized_features])
    
    return df_splits
    
def binarize_features(df_splits):
    # weather binarize
    for df in df_splits:
        df['weather2'] = df['weather'].apply(lambda w: int(w == 2))
        df['weather3'] = df['weather'].apply(lambda w: int(w == 3))
    ### Add more here if necessary
    return df_splits

def expand_features(df_splits):
    for df in df_splits:
        df['hour_sq'] = df['hour'].apply(lambda h: h**2)
        df['hour_cu'] = df['hour'].apply(lambda h: h**3)
        
        df['temp_inter'] = df['temp']*df['hightemp']
        df['wind_inter'] = df['windspeed']*df['highwind']
        df['hum_tem_inter'] = df['temp']*df['humidity']
        df['wind_week_inter'] = df['week']*df['windspeed']
        df['w2_temp_inter'] = df['weather2']*df['temp']
        df['w3_temp_inter'] = df['weather3']*df['temp']
    return df_splits
