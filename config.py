import ranking 

# unimportant features for splitting AND predicting = [atemp, season, datetime]
# binarized features = [weather]
# splitting features = [segment, holiday, workingday]
non_features = ['index', 'atemp', 'registered', 'casual', 'count', 'segment','datetime','holiday', 'workingday', 'season', 'weather']
non_features_counts = ['index', 'atemp','segment','datetime','holiday', 'workingday', 'season', 'weather']
filter = ranking.rf_ranking
folds = 2
split_vars = 'size_of_split, segment, holiday, workingday'

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