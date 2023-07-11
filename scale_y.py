import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import sklearn

import seaborn as sns
import pandas as pd
import numpy as np
import random

def normalize_y(df,dimension):
    print('Normalizing')
    print(df.shape)
    new_sample_df=pd.DataFrame()
    df=df.reset_index()
    run_ids=df[['algorithm_name','problem_id','instance_id','seed']].drop_duplicates().values
    df=df.set_index(['algorithm_name','problem_id','instance_id','seed','iteration'])
    df=df.sort_index()
    #x_columns = [f'x_{i}' for i in range(0,df.shape[1]-1)]
    #df.columns=x_columns + ['y']
    for algorithm_name,problem_id,instance_id,seed in run_ids:
        min_max_scaler = MinMaxScaler()
        trajectory_scaled=df.loc[(algorithm_name,problem_id,instance_id,seed,)].copy()
        y_scaled = min_max_scaler.fit_transform(trajectory_scaled[str(dimension)].values.reshape(-1, 1))
        trajectory_scaled[str(dimension)]=y_scaled
        trajectory_scaled[['algorithm_name','problem_id','instance_id','seed']]=algorithm_name,problem_id,instance_id,seed

        new_sample_df=pd.concat([new_sample_df,trajectory_scaled.reset_index(drop=False)])

    new_sample_df=new_sample_df.set_index(['algorithm_name','problem_id','instance_id','seed','iteration'])


    return new_sample_df

for dimension in [5]:
    for end_iteration in [20]:
        for algorithm in ['DE','ES','PSO']:
            for seed in [400,200,600,800,1000]:
                df=pd.read_csv(f'algorithm_run_data/{algorithm}_dim_{dimension}_seed_{seed}.csv',index_col=0)
                print(dimension,end_iteration,algorithm,seed)
                df['seed']=seed
                df=df.query("iteration<@end_iteration")
                df=normalize_y(df,dimension)
                df.to_csv(f'algorithm_run_data_normalized/{algorithm}_dim_{dimension}_seed_{seed}_end_iteration_{end_iteration}.csv')