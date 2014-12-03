import pandas as pd
import math
import datetime
import data_extraction as de
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

def discrete_vs_count(df, field, count):
    '''
    All of the features have a f airly small set of possible values
    df_train = pd.read_csv('../train.csv', header=0)
    de.discrete_vs_count(df_train, 'time')
    '''
    counts = []  
    vals =sorted(df[field].unique())
    for i in vals:
        c = df[count][df[field] == i]
        counts.append(c.sum()/len(c))
    return vals, counts


def histograms(df):
    df = df.copy()
    df = df.drop('datetime',1)
    df = df.drop('index',1)
    df.hist()
    for i in df.columns:
        plt.figure()
        plt.title(i)
        df[i].hist(bins= min(len(df[i].unique()), 50))

def variable_tables(df):
    df = df.copy()
    df = df.drop('datetime',1)
    df = df.drop('index',1)
    cat_file = open('cat-info.csv', 'w')
    cont_file = open('cont-info.csv', 'w')
    categorical = ['season', 'holiday', 'workingday', 'weather']#,'weekday']
    continous = ['temp', 'atemp', 'time', 'humidity', 'windspeed', 'casual' ,
                 'registered', 'count']
                 
    cat_header = 'Variable Name, values \n'
    cat_file.write(cat_header)
    for i in categorical:
        cat_file.write(i + ',' + convert_list(df[i].unique()) + '\n')
        
    cont_header = 'Variable Name, unique values, mean, std \n'
    cont_file.write(cont_header)
    for i in continous:
        cont_file.write(i + ',' + str(len(df[i].unique())) + ',' + str(df[i].mean()) 
                        + ',' +  str(df[i].std()) + '\n')
    
def convert_list(l):
    s = ''
    for i in l:
        s += str(i) + ';'
    return s[:-1]
    
def visualize(df, count):
    '''
    Produces a plot of count vs each of the 9 features.
    '''
    df_train = df.copy()
    #df_train['hour'] = df_train['datetime'].apply(de.create_hour)
    #df_train['week'] = df_train['datetime'].apply(de.create_week)
    features = ['hour', 'week', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    
    f, axarr = plt.subplots(3, 3)
    for i in xrange(3):
        for j in xrange(3):
            index = 3*i + j
            x,y = discrete_vs_count(df_train, features[index], count)
            axarr[i,j].scatter(x,y)
            axarr[i,j].plot(x,y)
            axarr[i,j].set_title(str(count)+ " vs " + features[index]) 
    plt.show()

def weekly_temp(df):
    weeks = list(df['week'].unique())
    temps = [0]*len(weeks)
    wind = [0]*len(weeks)
    humidity = [0]*len(weeks)
    for i in range(len(weeks)):
        temps[i] = np.mean(df[df['week'] == weeks[i]]['temp'])
        wind[i] = np.mean(df[df['week'] == weeks[i]]['windspeed'])
        humidity[i] = np.mean(df[df['week'] == weeks[i]]['humidity'])
    
    f,arr = plt.subplots()
    arr.scatter(weeks,humidity)
    arr.plot(weeks,humidity)
    arr.set_title('Humidity by Week')
    plt.show()
        
        

def plot3d(df, var_list, user, ax):
    x = df[var_list[0]].values
    y = df[var_list[1]].values
    x,y = np.meshgrid(x,y)
    z = np.zeros(x.shape)
    for i in xrange(x.shape[0]):
        for j in xrange(x.shape[1]):
            z[i,j] = get_knn_value(i, j, x, y, df, var_list, user)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)

def heatmap(df):
    rows = list(df['dow'].unique())
    columns = list(df['hour'].unique())
    
    # Get Data Counts
    registered_counts = np.zeros((len(rows),len(columns)))
    casual_counts = np.zeros((len(rows),len(columns)))
    casual_holiday_counts = np.zeros((1,len(columns)))
    registered_holiday_counts = np.zeros((1,len(columns)))
    for j in columns:
        for i in rows:
            registered_counts[i][j] = np.mean(df[(df['dow'] == i) & (df['holiday'] == 0)
                                & (df['hour'] == j)]['registered'])
            casual_counts[i][j] = np.mean(df[(df['dow'] == i) & (df['holiday'] == 0)
                                & (df['hour'] == j)]['casual'])
        casual_holiday_counts[0][j] = np.mean(df[(df['hour'] == j) 
                                            & (df['holiday'] == 1)]['casual'])
        registered_holiday_counts[0][j] = np.mean(df[(df['hour'] == j)
                                        & (df['holiday'] == 1)]['registered'])
    #holiday_counts = np.vstack((registered_holiday_counts,casual_holiday_counts))   
    registered_counts = np.vstack((registered_counts, registered_holiday_counts))
    casual_counts = np.vstack((casual_counts, casual_holiday_counts))                
    row_names = ['M', 'T', 'W', 'Th', 'F', 'Sat', 'Sun', 'Holiday']
    
    # Plot Regular Counts
    fig,ax=plt.subplots()
    ax.pcolor(registered_counts,cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(0,len(columns))+0.5)
    ax.set_yticks(np.arange(0,len(rows) + 1)+0.5)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.set_xticklabels(columns,minor=False,fontsize=10)
    ax.set_yticklabels(row_names,minor=False,fontsize=10)
    plt.text(0.5,1.06,'Registered Counts by Hour and Day of Week',
            fontsize=20,
            horizontalalignment='center',
            transform=ax.transAxes
            )
    plt.ylabel('Day of the Week',fontsize=15)
    plt.xlabel('Hour',fontsize=15)
    ax.set_xlim(0,24)
    plt.show()
    
    # holiday vs hour
    #row_names = ['Registered','Casual']
    #fig,ax=plt.subplots()
    #ax.pcolor(holiday_counts,cmap=plt.cm.Blues)
    #ax.set_xticks(np.arange(0,len(columns))+0.5)
    #ax.set_yticks(np.arange(0,2)+0.5)
    #ax.xaxis.tick_top()
    #ax.set_xticklabels(columns,minor=False,fontsize=10)
    #ax.set_yticklabels(row_names,minor=False,fontsize=10)
    #plt.text(0.5,1.06,'Holiday Counts by Hour',
    #        fontsize=20,
    #        horizontalalignment='center',
    #        transform=ax.transAxes
    #        )
    #plt.ylabel('User Type',fontsize=15)
    #plt.xlabel('Hour',fontsize=15)
    #ax.set_xlim(0,24)
    #plt.show()
    
    # Variable Correlation
    #features = ['temp','atemp','humidity','windspeed']
    #
    #correlation = np.zeros((len(features),len(features)))
    #for i in range(len(features)):
    #    for j in range(len(features)):
    #        correlation[i][j] = df[features[i]].corr(df[features[j]])
    #
    #fig,ax=plt.subplots()
    #ax.pcolor(correlation,cmap=plt.cm.Blues)
    #ax.set_xticks(np.arange(0,len(features))+0.5)
    #ax.set_yticks(np.arange(0,len(features))+0.5)
    #ax.xaxis.tick_top()
    #ax.yaxis.tick_left()
    #ax.set_xticklabels(features,minor=False,fontsize=10)
    #ax.set_yticklabels(features,minor=False,fontsize=10)
    #plt.text(0.5,1.06,'Pairwise Feature Correlation',
    #        fontsize=20,
    #        horizontalalignment='center',
    #        transform=ax.transAxes
    #        )
    #plt.show()

def get_knn_value(df):
    pass
def plot_var_3d(df, var_list):
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
                ax = fig.add_subplot(4,4,count, projection = '3d')
                new_df = de.produce_subsets(df, [s], [holiday], [wd], [weather])
                plot3d(new_df, var_list, user, ax)
                count += 1

    plt.show()
                
