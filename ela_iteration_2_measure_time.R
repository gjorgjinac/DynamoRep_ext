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



parser<-function(algorithm, seed, dim_str, problem_id_str){
    
    path<-paste("algorithm_run_data/", algorithm, "_dim_", dim, "_seed_", seed, ".csv", sep="", collapse="")
    print(path)
    dim<-as.numeric(dim)
    problem_id<-as.numeric(problem_id)   
	data<-read.csv(path, header = TRUE, check.names = FALSE)

	M<-matrix(0,100*30,96)

	count<-0
    mean_time<-0
		for(instance_id in 1:2){
            
			for(iteration in 0:29){
                print("INSTANCE ID")
                print(instance_id)
                print("ITERATION")
                print(iteration)
                data_temp<-data[data$problem_id==problem_id & data$instance_id==instance_id & data$iteration==iteration,]

                X<-subset(data_temp, select = c("0","1","2","3","4"))[,]
                y<-subset(data_temp, select = c("5"))[,]
                #print(X)
                #print(y)
                #features<-calculate_ELA(X,y)
                #M[count,]<-c(features,problem_id, instance_id, iteration)
                count<-count+1
                
                
                start_time <- Sys.time()

                # Run the code
                features<-calculate_ELA(X,y)

                # Get the current time after running the code
                end_time <- Sys.time()

                # Calculate the elapsed time
                time_taken <- end_time - start_time
                print(time_taken)
                mean_time<-mean_time+time_taken

			}
		}
	
    print('Mean time sum')
    print(mean_time)
    print('Count')
    print(count)
    print('Mean time')
    print(mean_time/count)
}


#path<-"../fixed/"
#path ga mapiramo na nextcloud

args = commandArgs(trailingOnly=TRUE)


algorithm<-args[1]
seed<-args[2]

dim<-args[3]
problem_id<-args[4]
          
parser(algorithm, seed, dim, problem_id)