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
    
def time_vs_count(df):
    counts = []
    for i in xrange(24):
        c = df['count'][df['time'] == i]
        counts.append(c.sum()/len(c))
    plt.plot(range(24), counts)

def visualize_date(df_train):
    return 0