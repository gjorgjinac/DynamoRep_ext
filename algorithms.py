import os
import random

import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.operators.sampling.lhs import LHS
from pymoo.problems import get_problem
from pymoo.optimize import minimize
import pandas as pd
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.core.problem import Problem
import sys

class COCOProblem(Problem):

    def __init__(self, name, n_var=10, pf_from_file=True, **kwargs):
        self.function, self.instance, self.object = get_bbob(name, n_var)
        self.name = name
        self.pf_from_file = pf_from_file

        coco = self.object
        n_var, n_obj, n_ieq_constr = coco.number_of_variables, coco.number_of_objectives, coco.number_of_constraints
        xl, xu = coco.lower_bounds, coco.upper_bounds

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_ieq_constr=n_ieq_constr,
                         xl=xl,
                         xu=xu,
                         **kwargs)

    def _calc_pareto_set(self, *args, **kwargs):
        if self.n_obj == 1:
            fname = '._bbob_problem_best_parameter.txt'

            self.object._best_parameter(what="print")
            ps = np.loadtxt(fname)
            os.remove(fname)

            return ps

    def _calc_pareto_front(self, *args, **kwargs):
        if self.pf_from_file:
            return Remote.get_instance().load("pymoo", "pf", "bbob.pf", to="json")[str(self.function)][str(self.instance)]
        else:
            ps = self.pareto_set()
            if ps is not None:
                return self.evaluate(ps)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.object(x) for x in X])

    def __getstate__(self):
        d = self.__dict__.copy()
        d["object"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.object = get_bbob(self.name, self.n_var)


def get_bbob(name, n_var=10, **kwargs):
    try:
        import cocoex as ex
    except:
        raise Exception("COCO test suite not found. \nInstallation Guide: https://github.com/numbbo/coco")

    args = name.split("-")
    suite = args[0]
    n_instance = int(args[-1])
    n_function = int(args[-2].replace("f", ""))

    assert 1 <= n_function <= 24, f"BBOB has 24 different functions to be chosen. {n_function} is out of range."

    suite_filter_options = f"function_indices: {n_function} " \
                           f"instance_indices: {n_instance} " \
                           f"dimensions: {n_var}"

    problems = ex.Suite(suite, "instances: 1-999", suite_filter_options)
    assert len(problems) == 1, "COCO problem not found."

    coco = problems.next_problem()

    return n_function, n_instance, coco


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)

def merge_generated_files(generated_trajectory_files, result_file):
    all_populations_df = pd.DataFrame()
    for file in generated_trajectory_files:
        df = pd.read_csv(file, index_col=[0])
        all_populations_df = pd.concat([all_populations_df, df])
    all_populations_df.to_csv(result_file)
    for file in generated_trajectory_files:
        os.remove(file)




def run(seed, dimension, instance_count=100):

    set_random_seed(seed)
    maximum_generations=50
    population_size = 10*dimension
    sampling = LHS()
    ga = GA(
        pop_size=population_size,
        sampling=sampling)

    de = DE(
        pop_size=population_size,seed=seed, sampling=sampling
    )

    n_offsprings = max(population_size*2, 200)
    es = ES( pop_size=population_size, seed=seed, sampling=sampling, n_offsprings=n_offsprings)
    nm = NelderMead()
    ps = PatternSearch()
    pso = PSO(pop_size=population_size, sampling=sampling,seed=seed)
    cmaes=CMAES(popsize=population_size,seed=seed, sampling=sampling)
    algorithms = {'ES':es,'DE':de,'PSO':pso}
    #algorithms = init_de_configurations(dimension, seed)
    algorithms_to_run = algorithms.items()
    result_dir = f'algorithm_run_data_popsize_50d_generations_20'
    os.makedirs(result_dir, exist_ok=True)

    for algorithm_name, algorithm in algorithms_to_run :
        generated_trajectory_files=[]
        result_file = os.path.join(result_dir,
                               f'{algorithm_name}_dim_{dimension}_seed_{seed}.csv')
        all_populations_df = pd.DataFrame()

        for problem_id in range(1,25):

            x_y_columns = [f'x_{i}' for i in range(0, dimension)] + ['y']

            for instance_id in range(1, instance_count+1):
                problem_name = f'bbob-{problem_id}-{instance_id}'
                problem = COCOProblem(problem_name, n_var=dimension)
                #cmaes = CMAES(x0=sampling, seed=seed)



                algorithm.termination = DefaultSingleObjectiveTermination()
                res = minimize(problem, termination=MaximumGenerationTermination(maximum_generations),
                               algorithm=algorithm, save_history=True,
                               seed=seed,
                               verbose=False)
                print(algorithm_name, problem_name)
               
                for iteration_index, iteration in enumerate(res.history):
                    pop_x = []
                    for population_individual in iteration.pop:
                        pop_x.append(list(population_individual.X) + list(population_individual.F))

                    population_df = pd.DataFrame(pop_x)
                    population_df['iteration'] = iteration_index
                    population_df['algorithm_name'] = algorithm_name
                    population_df['problem_id'] = problem_id
                    population_df['instance_id'] = instance_id
                    all_populations_df=all_populations_df.append(population_df)

                if instance_id%10==0:
                    all_populations_df.to_csv(result_file.replace('.csv', f'_{problem_id}_{instance_id}.csv'))
                    generated_trajectory_files+=[result_file.replace('.csv', f'_{problem_id}_{instance_id}.csv')]
                    all_populations_df = pd.DataFrame()



        merge_generated_files(generated_trajectory_files, result_file)

if __name__ == '__main__':

    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    arguments = sys.argv
    seed, dimension = int(arguments[1]), int(arguments[2])
    run(seed,dimension, instance_count=100)