import pandas as pd
import datetime
import matplotlib.pyplot as plt

def run():
    df_train = pd.read_csv('../train.csv', header=0)
    df_test = pd.read_csv('../test.csv', header=0)
    df_train['time'] = df_train['datetime'].apply(convert_date)
    df_test['time'] = df_test['datetime'].apply(convert_date)
    



def convert_date(date):
    d = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    time = d.hour + d.minute/60.0 + d.second/3600.0
    return time
    
def discrete_vs_count(df, field):
    '''
    All of the features have a fairly small set of possible values
    df_train = pd.read_csv('../train.csv', header=0)
    de.discrete_vs_count(df_train, 'time')
    '''
    counts = []  
    vals =sorted(df[field].unique())
    for i in vals:
        c = df['count'][df[field] == i]
        counts.append(c.sum()/len(c))
    return vals, counts
def weather_vs_count(df):
    vals = df['weather'].unique()

if __name__ == '__main__':
    df_train = pd.read_csv('../train.csv', header=0)
    df_train['time'] = df_train['datetime'].apply(convert_date)
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    f, axarr = plt.subplots(3, 3)
    for i in xrange(3):
        for j in xrange(3):
            index = 3*i + j
            x,y = discrete_vs_count(df_train, features[index])
            axarr[i,j].scatter(x,y)
            axarr[i,j].plot(x,y)
            axarr[i,j].set_title("counts vs " + features[index]) 
    plt.show()
  