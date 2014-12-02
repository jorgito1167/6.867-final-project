import pandas as pd
from sklearn import feature_selection
from sklearn import ensemble

#################
# K-best methods
#################
def rf_ranking(X,y):
    '''
    Random Forest Ranking
    '''
    # Change parameters later if necessary
    rf = ensemble.RandomForestRegressor(n_estimators = 50)
    rf.fit(X,y)
    importances = rf.feature_importances_
    return importances, [1./i for i in importances] 

def dt_ranking(X,y):
    '''
    Decision Tree ranking
    '''
    # Change parameters later if necessary
    dt = ensemble.RandomForestRegressor(n_estimators = 50)
    dt.fit(X,y)
    importances = dt.feature_importances_
    return importances, [1./i for i in importances] 


    