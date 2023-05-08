#!/usr/bin/env Rscript

#library(reticulate)
library(flacco)
library(data.table)

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



parser<-function(path, dimension){
	data<-read.table(gzfile(path))
    dimension<-as.numeric (dimension)
    print(dimension)
	problems<-seq(1,24,1)
	instances<-seq(1,999,1)
	itteration<-seq(0,49,1)
	M<-matrix(0,24*999*50,96)
	prob<-c()
	instance<-c()
	itter<-c()
	count<-1
	for(i in 1:length(problems)){
		for(j in 1:length(instances)){
			for(k in 1:length(itteration)){
			data_temp<-data[data$problem_id==problems[i] & data$instance_id==instances[j] & data$iteration==itteration[k],]
            print(data_temp)
			prob[count]<-unique(problems[i])
			instance[count]<-unique(instances[j])
			itter[count]<-itteration[k]
			X<-data_temp[,1:dimension]
			y<-data_temp[,dimension+1]
			features<-calculate_ELA(X,y)
			M[count,]<-c(features,prob[count],instance[count],itter[count])
			count<-count+1
			}
		}
	}
	colnames(M)<-c(names(features),"problem_id","instance_id","iteration_id")
	path_temp<-strsplit(path,".csv")[[1]][1]
	output_file<-paste("ela_trajectory/", path_temp,"_","ELA.csv",sep="",collapse="")
	write.csv(M,output_file)
}


#path<-"../fixed/"
args = commandArgs(trailingOnly=TRUE)
file = args[1]
dimension = args[2]
print(file)
print(dimension)
parser(file, dimension)