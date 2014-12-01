import numpy as np
from sklearn import linear_model
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import svm

'''
Each train method takes in the training dataframe and
uses the data to output a trained model with tuned 
hyperparameters. Cross-validation, tuning of hyperparameter
and feature creation should happen in the function.
'''

def train_k_nearest_neighbors(df_train):
    
    ##Feature Selection
    ## Model Training
    ## Cross Validation
    k = 5
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    x = df_train[features].values
    y = df_train['count'].values
    model = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance', 
                                    algorithm='kd_tree')
    model.fit(x,y)
    return model
                                    
def train_ridge_regression(df_test, df_train):
    features = ['time', 'season', 'holiday', 'workingday', 'weather', 'temp',
                'atemp', 'humidity', 'windspeed']
    x = df_train[features].values
    y = df_train['count'].values
    model = linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, 
    copy_X=True, max_iter=None, tol=0.001, solver='auto')
    
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