import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from utils import *
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold

#algorithm train_on_seed difference iterations dimension
arguments=sys.argv
argument_algorithm_names = arguments[1]
algorithm_names = get_argument_elements_from_list(arguments[1], False)
train_on_seed= True if arguments[2].lower()=='true' else False
difference= True if arguments[3].lower()=='true' else False

iteration_min, iteration_max = get_argument_elements_from_list(arguments[4],True)
seeds = [200,400,600,800,1000]

instance_min,instance_max=0,100
dimension=int(arguments[5])
normalize_y=True if arguments[6].lower()=='true' else False
result_dir=f'problem_classification_results_normalize_{normalize_y}'
os.makedirs(result_dir, exist_ok=True)
instance_count=instance_max-instance_min if instance_max!=999 else 1000


if False:
    to_run=[]
    for statistic in ['x','y']:
        for train_seed in seeds:

            if 'config' not in argument_algorithm_names:
                global_name=f'stat_{statistic}_dim_{dimension}_{"_".join(algorithm_names)}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_{"train" if train_on_seed else "test"}_on_seed_{train_seed}' + ('_differenced' if difference else '')
            else:
                global_name=f'stat_{statistic}_dim_{dimension}_{argument_algorithm_names}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_{"train" if train_on_seed else "test"}_on_seed_{train_seed}'+ ('_differenced' if difference else '')

            for fold in range(0,10):
                run_name=f'{global_name}_fold_{fold}'
                report_location=f'{result_dir}/{run_name}_report.csv'
                if not os.path.isfile(report_location):
                    to_run+=[report_location]

    if len(to_run)==0:
        print('Nothing to run')
        exit()




feature_df_file=f'{result_dir}/dim_{dimension}_{"-".join(algorithm_names)}_seeds_{"-".join([str(s) for s in seeds])}_it_{iteration_min}-{iteration_max}'
if os.path.isfile(feature_df_file):
    feature_df=pd.read_csv(feature_df_file, index_col=[0,1,2,3])
    print('read features from file')
else:
    sample_df=read_trajectory_data(algorithm_names,seeds,dimension)
    sample_df=sample_df.query('iteration>=@iteration_min and iteration<=@iteration_max')
    sample_df=sample_df.query('instance_id>=@instance_min and instance_id<=@instance_max')
    if normalize_y:
        sample_df=normalize(sample_df, dimension=dimension)
    feature_df = extract_features(sample_df, dimension, iteration_min, iteration_max, 'problem_classification')


    feature_df.to_csv(feature_df_file)





x_columns=list(filter(lambda c: np.any([f'_dim_{d}' in c or f'_x_{d}_' in c for d in range(0,dimension)]), feature_df.columns))
y_columns = list(filter(lambda c: f'_dim_{dimension}' in c or '_y' in c, feature_df.columns))
x_columns= list(set(x_columns).difference(set(y_columns)))
print(x_columns)
print(y_columns)

for statistic in ['x','y']:
    for train_seed in seeds:

        if 'config' not in argument_algorithm_names:
            global_name=f'stat_{statistic}_dim_{dimension}_{"_".join(algorithm_names)}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_{"train" if train_on_seed else "test"}_on_seed_{train_seed}' + ('_differenced' if difference else '')
        else:
            global_name=f'stat_{statistic}_dim_{dimension}_{argument_algorithm_names}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_{"train" if train_on_seed else "test"}_on_seed_{train_seed}'+ ('_differenced' if difference else '')

        kf = KFold(n_splits=10)
        fold=0
        instance_ids=np.array(feature_df.reset_index()['instance_id'].drop_duplicates())
        all_predictions=[]
        all_ys=[]
        accuracies=[]
        importances=[]
        
        if statistic =='x':
            stat_columns=x_columns
        else:
            stat_columns=y_columns
        print(statistic)
        print(stat_columns)
        stat_feature_df=feature_df[stat_columns+ ['y']]
        feature_names=stat_columns
        
        if difference:
            stat_feature_df=difference_features(stat_feature_df, iteration_max)
            feature_names=list(filter(lambda x: x!='y', stat_feature_df.columns))

        for train_index, test_index in kf.split(instance_ids):

            run_name=f'{global_name}_fold_{fold}'
            report_location=f'{result_dir}/{run_name}_report.csv'
            if os.path.isfile(report_location) and False:
                fold+=1
                print(f'Report already exists: {report_location}. Skipping run')
                continue
            train,test=get_split_data_for_problem_classification_generalization_testing(instance_ids, train_index, test_index, stat_feature_df, result_dir, run_name, train_seed, train_on_seed)

            clf, train, test= train_random_forest(train,test)
            preds,report_dict=save_classification_report(clf,test,report_location)

            #save_feature_importance(run_name, clf, dimension, iteration_min, iteration_max, result_dir,feature_names)

            test_predictions=pd.DataFrame(list(zip(test['y'].values,preds)), index=test.index, columns=['y','preds'])
            test_predictions.to_csv(f'{result_dir}/{run_name}_test_preds.csv', compression='zip')
            feature_importance_df=pd.DataFrame(list(clf.feature_importances_)).T
            feature_importance_df.columns=feature_names
            feature_importance_df.to_csv(f'{result_dir}/{run_name}_feature_importance.csv', compression='zip')

            fold+=1