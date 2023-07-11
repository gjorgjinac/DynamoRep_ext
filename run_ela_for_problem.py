import pandas as pd
from pflacco.classical_ela_features import *
import sys

arguments=sys.argv
algorithm=arguments[1]
dim=int(arguments[2])
seed=int(arguments[3])
problem_id=int(arguments[4])

d = pd.read_csv(f'algorithm_run_data/{algorithm}_dim_{dim}_seed_{seed}.csv', compression='zip',index_col=0)


ela_functions_x_y_parameters = [calculate_dispersion, calculate_ela_distribution, calculate_ela_level, calculate_ela_meta, calculate_information_content, calculate_nbc, calculate_pca]
ela_functions_x_y_lower_bound_upper_bound_parameters = [calculate_cm_angle, calculate_cm_conv, calculate_cm_grad, calculate_limo] 
ela_functions_x_y_f_parameters = [calculate_ela_conv]
                                  
ela_functions_x_y_lower_bound_upper_bound_f_dim_parameters = [calculate_ela_curvate, calculate_ela_local]


all_ela_features=[]
problem_samples=d.query("problem_id==@problem_id and algorithm_name==@algorithm")
for instance_id in range(1,101):
    instance_samples=problem_samples.query("instance_id==@instance_id")
    for iteration in range(0,50):
        iteration_samples=instance_samples.query("iteration==@iteration")
        X = iteration_samples[[str(i) for i in range(0,dim)]]
        Y = iteration_samples[str(dim)]
        iteration_ela_features={}
        for ela_function in ela_functions_x_y_parameters:
            try:
                ela_features=ela_function(X, Y)
                iteration_ela_features.update(ela_features)
            except Exception as e:
                print(str(ela_function))
                print(e)

        for ela_function in ela_functions_x_y_lower_bound_upper_bound_parameters:
            try:
                ela_features=ela_function(X, Y,-5,5)
                iteration_ela_features.update(ela_features)
            except Exception as e:
                print(str(ela_function))
                print(e)
        iteration_ela_features['problem_id']=problem_id
        iteration_ela_features['instance_id']=instance_id
        iteration_ela_features['algorithm_name']=algorithm
        iteration_ela_features['iteration']=iteration
        iteration_ela_features['seed']=seed
        all_ela_features+=[iteration_ela_features]
    if instance_id%10==0:
        pd.DataFrame(all_ela_features).to_csv(f'ela_features/{algorithm}_dim_{dim}_seed_{seed}_problem_id_{problem_id}.csv', compression='zip')
        
pd.DataFrame(all_ela_features).to_csv(f'ela_features/{algorithm}_dim_{dim}_seed_{seed}_problem_id_{problem_id}.csv', compression='zip')
                                                              