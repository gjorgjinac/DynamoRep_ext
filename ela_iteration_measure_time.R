#!/usr/bin/env Rscript
library(flacco)


calculate_ELA<-function(X,y){
	feat.object = createFeatureObject(X = X,y=y)
	ctrl = list(allow_cellmapping = FALSE,blacklist=c("pca","ela_distr", "ic","nbc"))
	features = calculateFeatures(feat.object, control = ctrl)
	
	features_pca <- NULL
	features_pca$pca.expl_var.cov_x <- NA;
	features_pca$pca.expl_var.cor_x <- NA;
	features_pca$pca.expl_var.cov_init <- NA;
	features_pca$pca.expl_var.cor_init <- NA;
	features_pca$pca.expl_var_PC1.cov_x <- NA;
	features_pca$pca.expl_var_PC1.cor_x <- NA;
	features_pca$pca.expl_var_PC1.cov_init <- NA;
	features_pca$pca.expl_var_PC1.cor_init <- NA;
	features_pca$pca.costs_fun_evals <- NA;
	features_pca$pca.costs_runtime <- NA;

	tryCatch({features_pca <-calculateFeatureSet(feat.object, set = "pca")},
		error = function(e){print("Error in PCA")}
	)

	
	features_ela_distr <- NULL
	features_ela_distr$ela_distr.skewness <- NA;
	features_ela_distr$ela_distr.kurtosis <- NA;
	features_ela_distr$ela_distr.number_of_peaks <- NA;
	features_ela_distr$ela_distr.costs_fun_evals <- NA;
	features_ela_distr$ela_distr.costs_runtime <- NA;

	tryCatch({features_ela_distr <-calculateFeatureSet(feat.object, set = "ela_distr")},
		error = function(x){print("Error in ela_distr");}
	)
	
	features_ic <- NULL
	features_ic$ic.h.max <- NA;
	features_ic$ic.eps.s <- NA;
	features_ic$ic.eps.ratio <- NA;
	features_ic$ic.m0 <- NA;
	features_ic$ic.costs_fun_evals <- NA;
	features_ic$ic.costs_runtime <- NA;
	features_ic$ic.eps.max <- NA;
	
	tryCatch(features_ic <-calculateFeatureSet(feat.object, set = "ic"),
		error = function(x){print("Error in ic");}
	)
			
	features_nbc <- NULL
	
	features_nbc$nbc.nn_nb.sd_ratio <- NA;
	
	features_nbc$nbc.nn_nb.mean_ratio <- NA;
	features_nbc$nbc.nn_nb.cor <- NA;
	features_nbc$nbc.dist_ratio.coeff_var <- NA;
	features_nbc$nbc.nb_fitness.cor <- NA;
	features_nbc$nbc.costs_fun_evals <- NA;
	features_nbc$nbc.costs_runtime <- NA;
	

	tryCatch({features_nbc <-calculateFeatureSet(feat.object, set = "nbc")},
		error = function(e){print("Error in NBC")}
	)
	


	features <- append(features, features_pca)
	features <- append(features, features_ela_distr)
	features <- append(features, features_ic)
	features <- append(features, features_nbc)
	
		
	temp_obj<-unlist(features, use.names=TRUE)
	return(temp_obj)
	
	
}



parser<-function(algorithm, seed, dim_str){
    
    path<-paste("algorithm_run_data/", algorithm, "_dim_", dim, "_seed_", seed, ".csv", sep="", collapse="")
    print(path)
    dim<-as.numeric(dim)
    
	data<-read.csv(path, header = TRUE, check.names = FALSE)

	M<-matrix(0,100*30,96)

	count<-0
    count_iteration<-0
    mean_time<-0
    sum_iteration<-0
    for(problem_id in 1:24){
		for(instance_id in 1:1){
            trajectory_time<-0
            count<-count+1
			for(iteration in 0:29){
                print("PROBLEM ID")
                print(problem_id)
                print("INSTANCE ID")
                print(instance_id)
                print("ITERATION")
                print(iteration)
                data_temp<-data[data$problem_id==problem_id & data$instance_id==instance_id & data$iteration==iteration,]
                print(paste0(0:(dim-1)))
                X <- subset(data_temp, select = paste0(0:(dim-1))) 
                
                
                y<-subset(data_temp, select = c(as.character(dim)))[,]

                #features<-calculate_ELA(X,y)
                #M[count,]<-c(features,problem_id, instance_id, iteration)
                
                
                count_iteration<-count_iteration+1
                start_time <- Sys.time()
                
                # Run the code
                features<-calculate_ELA(X,y)

                # Get the current time after running the code
                end_time <- Sys.time()

                # Calculate the elapsed time
                time_taken <- end_time - start_time
                print('Iteration time')
                print(time_taken)
                trajectory_time<-trajectory_time+time_taken
                sum_iteration<-sum_iteration+time_taken

			}
            print('Trajectory time')
            print(trajectory_time)
            mean_time<-mean_time+trajectory_time
		}
	}
    print('Sum time per trajectory')
    print(mean_time)
    print('Count trajectories')
    print(count)
    print('Mean time per trajectory')
    print(mean_time/count)
    
    
    print('Sum time per iteration')
    print(sum_iteration)
    print('Count iterations')
    print(count_iteration)
    print('Mean time per iteration')
    print(sum_iteration/count_iteration)
}


#path<-"../fixed/"
#path ga mapiramo na nextcloud

args = commandArgs(trailingOnly=TRUE)


algorithm<-args[1]
seed<-args[2]

dim<-args[3]

          
parser(algorithm, seed, dim)