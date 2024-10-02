import pandas as pd
from sklearn.model_selection import KFold
import os
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from utils import *
import sys

@timeitshort
def train_model(train, test, clf, target_columns=None):
    if target_columns is None:
        target_columns=['y']
    train_X=train.drop(columns=target_columns)
    train_X = train_X.fillna(train_X.mean())
    train_new=train_X.copy()
    train_new[target_columns]=train[target_columns]
    test_new = test.drop(columns=target_columns).fillna(train_X.mean())
    test_new[target_columns]=test[target_columns]
    clf.fit(train_new.drop(columns=target_columns), train_new[target_columns])
    return clf,train_new,test_new

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

def calculate_loss(y_true, y_pred,return_mean=True):
    if y_true.index.name is None or y_pred.index.name is None:
        y_true.index.name='index'
        y_pred.index.name='index'
    index_names=list(y_true.index.names)
    y_pred_new=y_pred.reset_index().melt(id_vars=index_names, value_vars=y_true.columns, var_name='algorithm_name', value_name='algorithm_score_predicted').sort_values(index_names+['algorithm_score_predicted']).reset_index(drop=True)

    y_pred_new['algorithm_rank']=[i%len(y_true.columns) for i in y_pred_new.index]

    y_true_new=y_true.reset_index().melt(id_vars=index_names, value_vars=y_true.columns, var_name='algorithm_name', value_name='algorithm_score_true').sort_values(index_names+['algorithm_score_true']).reset_index(drop=True)

    y_true_new['algorithm_rank']=[i%len(y_true.columns) for i in y_true_new.index]

    t=y_pred_new.merge(y_true_new,left_on=index_names+['algorithm_name'], right_on=index_names+['algorithm_name'], suffixes=['_predicted','_true'],)

    predicted_best=t.query('algorithm_rank_predicted==0').copy()[list(y_true.index.names) + ['algorithm_score_true']]
    true_best=t.query('algorithm_rank_true==0').copy()[list(y_true.index.names) + ['algorithm_score_true']]

    of_interest=predicted_best.merge(true_best, left_on=index_names, right_on=index_names, suffixes=['_predicted','_true'],)
    of_interest['score']=of_interest.apply(lambda x: 1- (x['algorithm_score_true_predicted']-x['algorithm_score_true_true']), axis=1)
    #of_interest['score']=of_interest.apply(lambda x: 1- (x['algorithm_score_true_predicted']), axis=1)
    return of_interest['score'].mean() if return_mean else of_interest[index_names+['score']]

def difference_ela_features(feature_df,max_iteration):
    columns_values=[]
    columns_names=[]
    for c_index, c in enumerate(feature_df.columns):
   
        iteration=int(c.split('_')[0])
        feature_name = "_".join(c.split('_')[1:])
        if iteration<=max_iteration-1:
            c_next=f'{iteration+1}_{feature_name}'
            columns_names+=[(f'{c_next} - {c}')]
            columns_values+=[(feature_df[c_next]- feature_df[c])]
     
    differenced_features=pd.concat(columns_values,axis=1)
    differenced_features.columns=columns_names
    #differenced_features['seed']=feature_df.reset_index()['seed'].values
    return differenced_features

df=pd.read_csv('../data/algorithm_run_data_unzipped/algorithm_run_data/all_algorithms_minimums_per_iteration.csv',index_col=0)
t=df.query('iteration==49').pivot(index=['problem_id','instance_id','dimension', 'seed'],columns= ['algorithm_name'], values=['min_up_to_iteration']).T
t_scaled=pd.DataFrame(MinMaxScaler().fit_transform(t), index=t.index, columns=t.columns)
performance=t_scaled.T.reset_index().groupby(['dimension','problem_id','instance_id']).mean().drop(columns=['seed'])
performance.columns=[c[1] for c in performance.columns]
print(performance)
seeds=[200,400,600,800,1000]
instance_min,instance_max=0,100




arguments=sys.argv
algorithm_names = get_argument_elements_from_list(arguments[1], False)
split_type = arguments[2]
difference= True if arguments[3].lower()=='true' else False
iteration_min,iteration_max= get_argument_elements_from_list(arguments[4],True)
dimension=int(arguments[5])
normalize_y=True if arguments[6].lower()=='true' else False
train_algorithm=arguments[7]




if normalize_y:
    feature_df=pd.read_csv(f'iteration_ela_normalized/{train_algorithm}_dim_{dimension}_all_runs_end_iteration_{iteration_max}.csv',index_col=[0,1,2,3,4], header=[0,1,2])
else:
    feature_df=pd.read_csv(f'iteration_ela/{train_algorithm}_dim_{dimension}_all_runs.csv',index_col=[0,1,2,3,4], header=[0,1,2])
print('Original feature df', feature_df.shape)


columns_from_iteraiton=list(filter(lambda x: x[1] in [str(e) for e in range(0,iteration_max+1)], feature_df.columns))
iteration_columns=list(filter(lambda x: x[1] in [str(e) for e in range(0,iteration_max+1)], feature_df.columns))
feature_df=feature_df[iteration_columns]
feature_df.columns=[f"{c[1]}_{c[2]}" for c in feature_df.columns]


if difference:
    feature_df=difference_ela_features(feature_df,iteration_max)
print(feature_df)
feature_names=feature_df.columns


feature_df=feature_df.reset_index().rename(columns={'algorithm':'algorithm_name'}).drop(columns=['dim'])

print(feature_df)



instance_count=instance_max-instance_min if instance_max!=999 else 1000
result_dir=f'AS_results_iteration_ela/normalize_{normalize_y}_difference_{difference}_split_{split_type}'
os.makedirs(result_dir,exist_ok=True)
feature_df_og=feature_df.copy()
if 'y' in feature_df.columns:
    feature_df=feature_df.drop(columns=['y'])
feature_df=feature_df_og.reset_index().merge(performance.loc[(dimension)], left_on=['problem_id','instance_id'],right_on=['problem_id','instance_id']).set_index(['algorithm_name','problem_id','instance_id','seed']).drop(columns=['index'])


all_losses=[]
df=feature_df.copy()
alg_df=df.loc[train_algorithm]


alg_res_dir=os.path.join(result_dir,f'dim_{dimension}_{"_".join(algorithm_names)}_it_{iteration_min}-{iteration_max}_instance_count_{instance_count}_trainalg_{train_algorithm}')
os.makedirs(alg_res_dir,exist_ok=True)

kf = KFold(10)

fold=0
problem_ids=np.array(feature_df.reset_index()['problem_id'].drop_duplicates())
instance_ids=np.array(feature_df.reset_index()['instance_id'].drop_duplicates())
ids = instance_ids if split_type=="I" else problem_ids
all_predictions=[]
all_ys=[]
accuracies=[]
importances=[]
id_columns=['problem_id','instance_id','algorithm_name','seed']
for train_index, test_index in kf.split(ids):



    train_ids, test_ids = ids [train_index], ids[test_index]
    print(train_ids)
    print(test_ids)
    if split_type=="I":
        train, test = alg_df.query('instance_id in @train_ids'), alg_df.query('instance_id in @test_ids')
    else:
        train, test = alg_df.query('problem_id in @train_ids'), alg_df.query('problem_id in @test_ids')
    train,test=preprocess_ela(train,test,id_columns)
    
    for model, model_name in [(RandomForestRegressor(),'rf'), (DummyRegressor(),'dummy')]:
        run_name=f'fold_{fold}_model_{model_name}'
        model, train, test= train_model(train,test,  model, target_columns=algorithm_names)
        print(train.columns)
        print(train.reset_index()[['seed','problem_id']].drop_duplicates())
        print(test.reset_index()[['seed','problem_id']].drop_duplicates())

        for split_data, split_name in [(train,'train'),(test,'test')]:
            preds=pd.DataFrame(model.predict(split_data.drop(columns=algorithm_names)), index=split_data.index, columns=algorithm_names)
            loss=calculate_loss(split_data[algorithm_names], preds)
            all_losses+=[(train_algorithm,fold,loss,model_name, split_name)]
            preds.columns=[f'pred_{c}' for c in preds.columns]
            pd.concat([split_data[algorithm_names],preds],axis=1).to_csv(f'{alg_res_dir}/{run_name}_{split_name}_preds.csv', compression='zip')
        if model_name=='rf':
            feature_importance_df=pd.DataFrame(list(model.feature_importances_)).T
            print(feature_importance_df)
            feature_importance_df.columns=model.feature_names_in_
            feature_importance_df.to_csv(f'{alg_res_dir}/{run_name}_feature_importance.csv', compression='zip')

    pd.DataFrame(all_losses, columns=['train_algorithm','fold','loss','model','split_name']).to_csv(f'{alg_res_dir}_all_results.csv')
    fold+=1


