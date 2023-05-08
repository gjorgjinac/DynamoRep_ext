import os
import sys
import math
import matplotlib
import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import KFold

#algorithm_names= ['DE','GA','ES']

#nohup python performance_prediction.py algorithm_name, difference, iterations, predict_log_precision, dimension > performance_

arguments=sys.argv
argument_algorithm_names = arguments[1]

algorithm_names = get_argument_elements_from_list(arguments[1], False)

difference= True if arguments[2].lower()=='true' else False
iteration_min, iteration_max = get_argument_elements_from_list(arguments[3],True)
id_columns=['algorithm_name', 'problem_id', 'instance_id', 'seed']
predict_log_precision=True if arguments[4].lower()=='true' else False
seeds = [200,400,600,800,1000]
dimension=int(arguments[5])
instance_min,instance_max=0,99

result_dir='performance_prediction_results'
os.makedirs(result_dir,exist_ok=True)


feature_df_file=f'{result_dir}/dim_{dimension}_{"-".join(algorithm_names)}_seeds_{"-".join([str(s) for s in seeds])}_it_{iteration_min}-{iteration_max}_instance_{instance_min}-{instance_max}.csv'
if os.path.isfile(feature_df_file):
    feature_df=pd.read_csv(feature_df_file, index_col=[0,1,2,3])
    print('read features from file')
else:
    sample_df=read_trajectory_data(algorithm_names,seeds,dimension)
    sample_df=sample_df.query('instance_id>=@instance_min and instance_id<=@instance_max')
    feature_sample_df=sample_df.query('iteration>=@iteration_min and iteration<=@iteration_max')
    feature_df = extract_features(feature_sample_df, dimension, iteration_min, iteration_max)
    feature_df = feature_df.drop(columns=['y'])
    precision_df = get_precision(sample_df,dimension)
    feature_df = feature_df.merge(precision_df,left_on=id_columns,right_on=id_columns)
    feature_df=feature_df.rename(columns={'precision':'y'})
    feature_df=feature_df.set_index(id_columns)
    feature_df.to_csv(feature_df_file)
    
feature_df['y']=feature_df['y'].apply(lambda x: x if x != 0 else 0.00000001)

if predict_log_precision:
    feature_df['y']=feature_df['y'].apply(lambda x: math.log(x))
    

my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','#008b8b','#9acd32','#e3f8b7'])


if difference:
    feature_df=difference_features(feature_df,iteration_max)

feature_names=list(filter(lambda x: x!='y', feature_df.columns))

instance_count=instance_max-instance_min if instance_max!=999 else 1000


for train_seed in seeds:

    if 'config' not in argument_algorithm_names:
        global_name=f'dim_{dimension}_{"_".join(algorithm_names)}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_seed_{train_seed}_log_{predict_log_precision}' + ('_differenced' if difference else '')
    else:
        global_name=f'dim_{dimension}_{argument_algorithm_names}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_seed_{train_seed}_log_{predict_log_precision}'+ ('_differenced' if difference else '')

    kf = KFold(n_splits=10)
    df=feature_df.query("seed==@train_seed")
    fold=0
    problem_ids=np.array(feature_df.reset_index()['problem_id'].drop_duplicates())
    all_predictions=[]
    all_ys=[]
    accuracies=[]
    importances=[]

    for train_index, test_index in kf.split(problem_ids):

        run_name=f'{global_name}_fold_{fold}'
        predictions_location=f'{result_dir}/{run_name}_test_preds.csv'
        if os.path.isfile(predictions_location):
            print(f'Predictions already exists: {predictions_location}. Skipping run')
            fold+=1
            continue
        train,test=get_split_data_for_algorithm_classification(problem_ids,train_index, test_index, df, result_dir, run_name)
        print('training')
        clf, train, test= train_random_forest(train,test,do_regression=True)
        print('predicting')
        preds = clf.predict(test.drop(columns=['y']))
        test_predictions=pd.DataFrame(list(zip(test['y'].values,preds)), index=test.index, columns=['y','preds'])
        test_predictions.to_csv(predictions_location, compression='zip')
        feature_importance_df=pd.DataFrame(list(clf.feature_importances_)).T
        feature_importance_df.columns=feature_names
        feature_importance_df.to_csv(f'{result_dir}/{run_name}_feature_importance.csv', compression='zip')

        fold+=1
