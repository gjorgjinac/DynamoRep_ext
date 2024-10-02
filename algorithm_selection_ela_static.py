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
    train_X = train_X.replace([np.inf,-np.inf], np.nan).dropna(axis=1)
    train_new=train_X.copy()
    train_new[target_columns]=train[target_columns]
    test_new = test.drop(columns=target_columns)[train_X.columns].replace([np.inf,-np.inf], np.nan).dropna()
    test_new[target_columns]=test[target_columns]
    clf.fit(train_new.drop(columns=target_columns), train_new[target_columns])
    return clf,train_new,test_new


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
dimension=int(arguments[3])
normalize_y=True if arguments[4].lower()=='true' else False
sample_count_dimension_factor=int(arguments[5])


feature_df=pd.read_csv(f'static_ela_features/dim_{dimension}_{sample_count_dimension_factor}d{"_scaled" if normalize_y else ""}.csv')
feature_df['problem_id']=feature_df['f'].apply(lambda x: int(x.split('_')[0]))
feature_df['instance_id']=feature_df['f'].apply(lambda x: int(x.split('_')[1]))
feature_df=feature_df.drop(columns=['f'])




instance_count=instance_max-instance_min if instance_max!=999 else 1000
result_dir=f'AS_results_static_ela/normalize_{normalize_y}_split_{split_type}'
os.makedirs(result_dir,exist_ok=True)
alg_res_dir=os.path.join(result_dir,f'dim_{dimension}_{"_".join(algorithm_names)}_{sample_count_dimension_factor}d_instance_count_{instance_count}')
os.makedirs(alg_res_dir,exist_ok=True)



feature_df_og=feature_df.copy()
feature_df=feature_df_og.reset_index().merge(performance.loc[(dimension)], left_on=['problem_id','instance_id'],right_on=['problem_id','instance_id']).set_index(['problem_id','instance_id']).drop(columns=['index'])
print(feature_df)

kf = KFold(10)
fold=0
problem_ids=np.array(feature_df.reset_index()['problem_id'].drop_duplicates())
instance_ids=np.array(feature_df.reset_index()['instance_id'].drop_duplicates())
ids = instance_ids if split_type=="I" else problem_ids
all_losses=[]

for train_index, test_index in kf.split(ids):



    train_ids, test_ids = ids [train_index], ids[test_index]
    print(train_ids)
    print(test_ids)
    if split_type=="I":
        train, test = feature_df.query('instance_id in @train_ids'), feature_df.query('instance_id in @test_ids')
    else:
        train, test = feature_df.query('problem_id in @train_ids'),feature_df.query('problem_id in @test_ids')

    for model, model_name in [(RandomForestRegressor(),'rf'), (DummyRegressor(),'dummy')]:
        run_name=f'fold_{fold}_model_{model_name}'
        model, train, test= train_model(train,test,  model, target_columns=algorithm_names)
        print(train.reset_index()['problem_id'].drop_duplicates())
        print(test.reset_index()['problem_id'].drop_duplicates())

        for split_data, split_name in [(train,'train'),(test,'test')]:
            preds=pd.DataFrame(model.predict(split_data.drop(columns=algorithm_names)), index=split_data.index, columns=algorithm_names)
            loss=calculate_loss(split_data[algorithm_names], preds)
            all_losses+=[(fold,loss,model_name, split_name)]
            preds.columns=[f'pred_{c}' for c in preds.columns]
            pd.concat([split_data[algorithm_names],preds],axis=1).to_csv(f'{alg_res_dir}/{run_name}_{split_name}_preds.csv', compression='zip')
        if model_name=='rf':
            feature_importance_df=pd.DataFrame(list(model.feature_importances_)).T
            print(feature_importance_df)
            feature_importance_df.columns=model.feature_names_in_
            feature_importance_df.to_csv(f'{alg_res_dir}/{run_name}_feature_importance.csv', compression='zip')

    pd.DataFrame(all_losses, columns=['fold','loss','model','split_name']).to_csv(f'{alg_res_dir}_all_results.csv')
    fold+=1
