import os
import pandas as pd
import missingno as msno
from config import *
import sys


arguments=sys.argv
dim=int(arguments[1])

all_ela_features=pd.DataFrame()
for algorithm in ['ES','DE','PSO','CMAES']:
    print(algorithm)

    for seed in [200,400,600,800,1000]:
        ela=pd.DataFrame()
        for problem_id in range(1,25):
            problem_ela=pd.read_csv(f'ela_features/{algorithm}_dim_{dim}_seed_{seed}_problem_id_{problem_id}.csv', index_col=[0],compression='zip')
            ela = pd.concat([ela,problem_ela])

        ela['dim']=dim
        ela.to_csv(f'ela_features_merged/{algorithm}_dim_{dim}_seed_{seed}.csv', compression='zip')
        all_ela_features=pd.concat([all_ela_features,ela])
        print(ela.shape)
            
id_columns=['problem_id','instance_id','algorithm_name','iteration','dim','seed']
            
xs = []
ys = []
ids = []
a=all_ela_features.melt(id_vars=id_columns).pivot(index=['problem_id','instance_id','algorithm_name','dim','seed'], columns=['iteration','variable'])
a.to_csv(f'ela_features_merged/ELA_features_dim_{dim}_all_runs.csv')