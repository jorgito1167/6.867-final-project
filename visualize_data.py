import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def visualize():
    '''
    Produces a plot of count vs each of the 9 features.
    '''
    df_train = pd.read_csv('../train.csv', header=0)
    df_train['time'] = df_train['datetime'].apply(convert_date)
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    change_season(df_train)
    change_weather(df_train)
    f, axarr = plt.subplots(3, 3)
    for i in xrange(3):
        for j in xrange(3):
            index = 3*i + j
            x,y = discrete_vs_count(df_train, features[index])
            axarr[i,j].scatter(x,y)
            axarr[i,j].plot(x,y)
            axarr[i,j].set_title("counts vs " + features[index]) 
    plt.show()

def plot3d(var_list, df, ax):
    x = df[var_list[0]].values
    y = df[var_lsit[1]].values
    z = df[var_list[2]].values
    ax.plot_surface(x, y, z)

def plot_var_3d(var_list, df):
    wds = [0,1]
    user_type = [0,1]
    seasons = [1,2,3,4]
    holiday = 0
    weather = 1
    fig = plt.figure()
    count = 1
    for wd in wds:
        for user in user_type:
            for s in seasons:
                i = floor(count/4.0)
                j = count%4
                ax = fig.add_subplot(1,i,j, projection = '3d')
                new_df = produce_subsets(df, season, holiday, wd, weather)
                plot3d(var_list, ax)
                count += 1

    plot.show()
                
