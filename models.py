import numpy as np
from sklearn import linear_model
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
import config
import main as m
import timeit

'''
Each train method takes in the training dataframe and
uses the data to output a trained model with tuned 
hyperparameters. Cross-validation, tuning of hyperparameter
and feature creation should happen in the function.
'''

def train_k_nearest_neighbors(df, user_type):  
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='kd_tree')
    mod_knn = Pipeline([('filter', feat_filter), ('knn', model)])
    best_score = float('inf')
    for k in xrange(1,4):
        for n in xrange(5,10):
            print 'ok'
            mod_knn.set_params(filter__k= k, knn__n_neighbors= n)
            mod_knn.fit(x,y)
            print 'ok1'
            val_score = cross_val_score(mod_knn, x, y, scoring = 'log_loss', cv = 3)
            print 'ok2'
            if val_score < best_score:
                best_score = val_score
                best_k = k
                best_n = n
    mod_knn.set_params(filter__k= best_k, knn__n_neighbors= best_n)
    return mod_knn
                                    
def train_ridge_regression(df, user_type):
    print 'Start training'
    start = timeit.default_timer()
    if df.empty:
        return None
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model = linear_model.Ridge(alpha=1.0)
    mod_ridge = Pipeline([('filter', feat_filter), ('ridge', model)])
    best_score = float('inf')
    
    for k in xrange(1,4):
        for n in xrange(1,10):
            mod_ridge.set_params(filter__k= k, ridge__alpha= n)
            val_score = cross_val_score(mod_ridge, x, y, scoring = m.metric, cv = 8)
            mean_val_score = val_score.mean()
            print 'ok'
            print mean_val_score
            if mean_val_score < best_score:
                best_score = mean_val_score
                best_k = k
                best_n = n
    mod_ridge.set_params(filter__k= best_k, ridge__alpha= best_n)
    print "Size of split: " + str(len(y))
    print "Time to train: " + str(timeit.default_timer()-start)
    print 'Done training'
    return mod_ridge
    
    
def train_LASSO(df_test, df_train):
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    x = df_train[features].values
    y = df_train['count'].values
    model = sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, 
    precompute='auto', copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False)
    
def train_elastic_nets(df_test, df_train):
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    x = df_train[features].values
    y = df_train['count'].values
    model = linear_model.ElasticNetCV(n_jobs = 3)
    model.fit(x,y)
    
    x_test = df_test[features]
    df_test['count'] = model.predict(x_test)
    df_test['count'][ df_test['count']<0 ] = 0
    output = df_test[['datetime', 'count']]
    output.to_csv('out.csv', index = False)
    
def train_gaussian_processes(df_test, df_train):
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    x = df_train[features].values
    y = df_train['count'].values
    model = sklearn.gaussian_process.GaussianProcess(regr='constant', corr='squared_exponential', beta0=None, 
    storage_mode='full', verbose=False, theta0=0.1, thetaL=None, thetaU=None, optimizer='fmin_cobyla', 
    random_start=1, normalize=True, nugget=2.2204460492503131e-15, random_state=None)
    
def train_support_vector_regression(df_test, df_train):
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    x = df_train[features].values
    y = df_train['count'].values
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    x = df_train[features].values
    y = df_train['count'].values 
    model =  sklearn.svm.SVR(kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, 
     shrinking=True, probability=False, cache_size=200, verbose=False, max_iter=-1, random_state=None)