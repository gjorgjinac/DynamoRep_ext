import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from utils import *
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold

def preprocess_ela(train,test,id_columns):
    train=train.replace([np.inf, -np.inf], np.nan)
    test=test.replace([np.inf, -np.inf], np.nan)
    count_missing={column:train[column].isna().sum() for column in train.columns}
    count_missing=pd.DataFrame([count_missing]).T
    count_missing.columns=['missing']
    count_missing['missing_percent']=count_missing['missing'].apply(lambda x: x/train.shape[0])
    
    columns_to_keep=list(count_missing.query('missing_percent<0.1').index)
    print('Keeping columns', columns_to_keep)
    train=train[columns_to_keep]
    test=test[columns_to_keep]
    train = train.fillna(train.mean())
    test = test.fillna(train.mean())
    return train,test

arguments=sys.argv
argument_algorithm_names = arguments[1]
if argument_algorithm_names == 'de_config':
    algorithm_names = [f'DE_CR_{CR}_crossover_{crossover}' for CR in [0.2, 0.4, 0.6, 0.8] for crossover in
                       ['exp', 'bin']]
else:
    algorithm_names = get_argument_elements_from_list(arguments[1], False)
train_on_seed= True if arguments[2].lower()=='true' else False
difference= True if arguments[3].lower()=='true' else False
end_iteration=int(arguments[4])
dimension=int(arguments[5])
result_dir='algorithm_classification_ela_results'
seeds=[200,400,600,800,1000]
id_columns=['problem_id','instance_id','algorithm','dim','seed']
instance_min, instance_max=0,100
os.makedirs(result_dir, exist_ok=True)

iteration_min,iteration_max=0,end_iteration

feature_df=pd.DataFrame()
for algorithm in algorithm_names:
    feature_df=pd.concat([feature_df, pd.read_csv(f'iteration_ela/{algorithm}_dim_{dimension}_all_runs.csv',index_col=[0,1,2,3,4], header=[0,1,2])])

print(feature_df)
print('Original feature df', feature_df.shape)

iteration_columns=list(filter(lambda x: x[1] in [str(e) for e in range(0,end_iteration+1)], feature_df.columns))
feature_df=feature_df[iteration_columns]

print('Feature df from algorithm and iteration', feature_df.shape)

feature_df['y']=list(feature_df.reset_index()['algorithm'].values)
#feature_df=feature_df.set_index(id_columns)



my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','#008b8b','#9acd32','#e3f8b7'])


instance_count=instance_max-instance_min if instance_max!=999 else 1000


for train_seed in seeds:

    if 'config' not in argument_algorithm_names:
        global_name=f'dim_{dimension}_{"_".join(algorithm_names)}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_{"train" if train_on_seed else "test"}_on_seed_{train_seed}' + ('_differenced' if difference else '')
    else:
        global_name=f'dim_{dimension}_{argument_algorithm_names}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_{"train" if train_on_seed else "test"}_on_seed_{train_seed}'+ ('_differenced' if difference else '')

    kf = KFold(n_splits=10)
    fold=0
    problem_ids=np.array(feature_df.reset_index()['problem_id'].drop_duplicates())
    all_predictions=[]
    all_ys=[]
    accuracies=[]
    importances=[]

    for train_index, test_index in kf.split(problem_ids):

        run_name=f'{global_name}_fold_{fold}'
        report_location=f'{result_dir}/{run_name}_report.csv'
        if os.path.isfile(report_location):
            print(f'Report already exists: {report_location}. Skipping run')
            continue
        train,test=get_split_data_for_algorithm_classification_generalization_testing(problem_ids, train_index, test_index, feature_df, result_dir, run_name, train_seed, train_on_seed)
        print('Before preprocessing', train.shape)
        train,test=preprocess_ela(train,test,id_columns)
        print('After preprocessing', train.shape)
        
        train_X=train.drop(columns=[('y','','')])
        train_y=train[('y','','')]
        
        test_X=test.drop(columns=[('y','','')])
        test_y=test[('y','','')]
        
        feature_names=train_X.columns 
        clf = RandomForestClassifier()
        print(train.shape)
        print(train_y.shape)
        clf.fit(train_X, train_y)
        
        
        
        preds = clf.predict(test_X)
        #report_dict = classification_report(test[('y','','')], preds,  output_dict=True)
        #report_df = pd.DataFrame(report_dict)
        #report_df.to_csv(report_location)

        #save_feature_importance(run_name, clf, dimension, iteration_min, iteration_max, result_dir,feature_names)

        test_predictions=pd.DataFrame(list(zip(test_y.values,preds)), index=test.index, columns=['y','preds'])
        test_predictions.to_csv(f'{result_dir}/{run_name}_test_preds.csv', compression='zip')
        feature_importance_df=pd.DataFrame(list(clf.feature_importances_)).T
        feature_importance_df.columns=feature_names
        feature_importance_df.to_csv(f'{result_dir}/{run_name}_feature_importance.csv', compression='zip')

        fold+=1