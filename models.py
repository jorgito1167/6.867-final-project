import numpy as np
from sklearn import linear_model
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import svm
from sklearn import ensemble
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
    print 'training split'
    start = timeit.default_timer()
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model = neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='kd_tree')
    mod_knn = Pipeline([('filter', feat_filter), ('knn', model)])
    best_score = float('inf')
    for k in xrange(1,5):
        for n in xrange(1,7):
            mod_knn.set_params(filter__k= k, knn__n_neighbors= n)
            val_score = cross_val_score(mod_knn, x, y, scoring = m.metric, cv = config.folds, n_jobs = 3)
            mean_val_score = val_score.mean()
            if mean_val_score < best_score:
                best_score = mean_val_score
                best_v_score = val_score
                best_k = k
                best_n = n
    mod_knn.set_params(filter__k= best_k, knn__n_neighbors= best_n)
    mod_knn.fit(x,y)
    time_to_train = str(np.round(timeit.default_timer()-start,2))
    out_str = time_to_train + ',' + str(best_k) + ',' + str(best_n) + ','
    out_str += str(np.round(best_v_score,3).tolist())[1:-1] + ',' + str(best_score) + '\n'
    print "Size of split: " + str(len(y))
    print "Time to train: " + time_to_train
    return mod_knn, out_str
                                    
def train_ridge_regression(df, user_type):
  
    start = timeit.default_timer()
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model = linear_model.Ridge(alpha=1.0)
    mod_ridge = Pipeline([('filter', feat_filter), ('ridge', model)])
    best_score = float('inf')
    
    for k in xrange(1,5):
        for n in [1e-3, 1, 1e3]:
            mod_ridge.set_params(filter__k= k, ridge__alpha= n)
            val_score = cross_val_score(mod_ridge, x, y, scoring = m.metric, cv = config.folds, n_jobs = 3)
            mean_val_score = val_score.mean()
            if mean_val_score < best_score:
                best_score = mean_val_score
                best_v_score = val_score
                best_k = k
                best_n = n
    mod_ridge.set_params(filter__k= best_k, ridge__alpha= best_n)
    mod_ridge.fit(x,y)
    time_to_train = str(np.round(timeit.default_timer()-start,2))
    out_str = time_to_train + ',' + str(best_k) + ',' + str(best_n) + ','
    out_str += str(np.round(best_v_score,3).tolist())[1:-1] + ',' + str(best_score) + '\n'
    print "Size of split: " + str(len(y))
    print "Time to train: " + time_to_train
    return mod_ridge, out_str
    
    
def train_LASSO(df, user_type):
    start = timeit.default_timer()
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model = linear_model.Lasso(alpha=1.0, normalize=False)
    mod_lasso = Pipeline([('filter', feat_filter), ('lasso', model)])
    best_score = float('inf')
    
    for k in xrange(1,5):
        for n in [1e-3, 1, 1e3]:
            mod_lasso.set_params(filter__k= k, lasso__alpha= n)
            val_score = cross_val_score(mod_lasso, x, y, scoring = m.metric, cv = config.folds, n_jobs = 3)
            mean_val_score = val_score.mean()
            mean_val_score = val_score.mean()
            if mean_val_score < best_score:
                best_score = mean_val_score
                best_v_score = val_score
                best_k = k
                best_n = n
    mod_lasso.set_params(filter__k= best_k, lasso__alpha= best_n)
    mod_lasso.fit(x,y)
    time_to_train = str(np.round(timeit.default_timer()-start,2))
    out_str = time_to_train + ',' + str(best_k) + ',' + str(best_n) + ','
    out_str += str(np.round(best_v_score,3).tolist())[1:-1] + ',' + str(best_score) + '\n'
    print "Size of split: " + str(len(y))
    print "Time to train: " + time_to_train
    return mod_lasso, out_str
    
def train_elastic_nets(df, user_type):
    start = timeit.default_timer()
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model = linear_model.ElasticNet(alpha=1.0)
    mod_elastic = Pipeline([('filter', feat_filter), ('elastic', model)])
    best_score = float('inf')
    
    for k in xrange(1,5):
        for n in [1e-3, 1, 1e3]:
            mod_elastic.set_params(filter__k= k, elastic__alpha= n)
            val_score = cross_val_score(mod_elastic, x, y, scoring = m.metric, cv = config.folds, n_jobs = 3)
            mean_val_score = val_score.mean()
            mean_val_score = val_score.mean()
            if mean_val_score < best_score:
                best_score = mean_val_score
                best_v_score = val_score
                best_k = k
                best_n = n
    mod_elastic.set_params(filter__k= best_k, elastic__alpha= best_n)
    mod_elastic.fit(x,y)
    time_to_train = str(np.round(timeit.default_timer()-start,2))
    out_str = time_to_train + ',' + str(best_k) + ',' + str(best_n) + ','
    out_str += str(np.round(best_v_score,3).tolist())[1:-1] + ',' + str(best_score) + '\n'
    print "Size of split: " + str(len(y))
    print "Time to train: " + time_to_train
    return mod_elastic, out_str
    

    
def train_random_forest_regressor(df, user_type):
    start = timeit.default_timer()
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model = ensemble.RandomForestRegressor(n_estimators=100)
    mod = Pipeline([('filter', feat_filter), ('rand', model)])
    best_score = float('inf')    
    for k in xrange(1,5):
            mod.set_params(filter__k= k)
            val_score = cross_val_score(mod, x, y, scoring = m.metric, cv = config.folds, n_jobs = 3)
            mean_val_score = val_score.mean()
            mean_val_score = val_score.mean()
            if mean_val_score < best_score:
                best_score = mean_val_score
                best_v_score = val_score
                best_k = k
    mod.set_params(filter__k= best_k)
    time_to_train = str(np.round(timeit.default_timer()-start,2))
    out_str = time_to_train + ',' + str(best_k) + ',' + str(None) + ','
    out_str += str(np.round(best_v_score,3).tolist())[1:-1] + ',' + str(best_score) + '\n'
    print "Size of split: " + str(len(y))
    print "Time to train: " + time_to_train
    return mod, out_str
    
def train_support_vector_regression(df, user_type):
    start = timeit.default_timer()
    if df.empty:
        return None
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model =  svm.SVR(kernel='rbf', degree=3, gamma=0.0, C=1.0, epsilon=1)
    mod = Pipeline([('filter', feat_filter), ('svr', model)])
    best_score = float('inf')
    
    for k in xrange(1,5):
        for C in [1e-3, 1, 1e3]:
            for g in [1e-3, 1, 1e3]:
                mod.set_params(filter__k= k, svr__gamma= g, svr__C= C)
                val_score = cross_val_score(mod, x, y, scoring = m.metric, cv = config.folds, n_jobs = 3)
                mean_val_score = val_score.mean()
                mean_val_score = val_score.mean()
                if mean_val_score < best_score:
                    best_score = mean_val_score
                    best_v_score = val_score
                    best_k = k
                    best_g = g
                    best_C = C
    mod.set_params(filter__k= best_k, svr__gamma= best_g, svr__C= best_C)
    mod.fit(x,y)
    time_to_train = str(np.round(timeit.default_timer()-start,2))
    out_str = time_to_train + ',' + str(best_k) + ',' + str(best_C) + ','
    out_str += str(np.round(best_v_score,3).tolist())[1:-1] + ',' + str(best_score) + '\n'
    print "Size of split: " + str(len(y))
    print "Time to train: " + time_to_train
    return mod, out_str
    


def train_gaussian_processes(df, user_type):
    start = timeit.default_timer()
    if df.empty:
        return None
    x = df.drop(config.non_features, 1).values
    y = df['casual' if user_type else 'registered']
    
    feat_filter = SelectKBest(config.filter, k=1)
    model = gaussian_process.GaussianProcess(regr='constant', corr='squared_exponential')
    mod_elastic = Pipeline([('filter', feat_filter), ('elastic', model)])
    best_score = float('inf')
    
    for k in xrange(1,5):
        for n in [1e-3, 1, 1e3]:
            mod_elastic.set_params(filter__k= k, elastic__alpha= n)
            val_score = cross_val_score(mod_elastic, x, y, scoring = m.metric, cv = config.folds, n_jobs = 3)
            mean_val_score = val_score.mean()
            if mean_val_score < best_score:
                best_score = mean_val_score
                best_k = k
                best_n = n
    mod_elastic.set_params(filter__k= best_k, elastic__alpha= best_n)
    print "Size of split: " + str(len(y))
    print "Time to train: " + str(timeit.default_timer()-start)
    return mod_elastic
    model = gaussian_process.GaussianProcess(regr='constant', corr='squared_exponential')
