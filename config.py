import ranking 

non_features = ['index', 'registered', 'casual', 'count', 'datetime','holiday', 'workingday', 'season', 'weather']
non_features_counts = ['index', 'datetime','holiday', 'workingday', 'season', 'weather']
filter = ranking.rf_ranking
folds = 2
split_vars = 'size_of_split, season, holiday, workingday, weather'

def split_variables():
    split_variables = 'split number, count_type,' + split_vars +',time_to_train, best_k, best_alpha'
    for i in xrange(folds):
        split_variables += ',error ' + str(i)
    split_variables +=',mean error\n'
    return split_variables
    
def get_vars(df):
    v = split_vars.replace(' ', '').split(',')
    out = str(df.shape[0]) + ','
    for i in v[1:]:
        out += str(df[i].unique()[0]) +','
    return out