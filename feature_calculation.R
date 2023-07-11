# Run this command in Base environment
# R -e 'install.packages("flacco", dependencies = TRUE, repos = "http://cran.us.r-project.org", Ncpus = -1)'

library(flacco)

calculate_features <- function(X, Y) {

  dim = ncol(X)
  samples = nrow(X)
  samples_per_cell= 10
  #block = round((samples/10)^(1/dim))

  feat.object <- createFeatureObject(X = X, y = Y)

  features = c()
    
  ela_types <- c("cm_angle", "cm_conv", "cm_grad", "ela_distr", "ela_level", "ela_meta", "basic", "disp", "limo", "nbc", "pca", "bt", "gcm", "ic")
    
  # Needs sampling
  # "ela_conv", "ela_curv", "ela_local", "ela_conv"

  for (ela_type in ela_types) {

      tryCatch(
        expr = {
          fe = calculateFeatureSet(feat.object, ela_type)
          features <- c(features,fe)
        },
        error = function(w){
            print(paste("Can not compute", ela_type, sep=" "))
        }
      )
      
  }

  return(features)
}

args = commandArgs(trailingOnly=TRUE)
file = args[1]

df=read.csv(paste(file, "csv", sep="."));
nrow(df)

datalist = list()

numclasses = 24
numinstances = 100

write_file=paste(file, "ela", "csv", sep=".");

if (file.exists(write_file))
{
    print("File already exists. Skipping computing ELA features");
} else {
    for (class in 1:numclasses) {

      for (instance in 1:numinstances) {

        sdf = df[df[, 3]==instance & df[, 2]==class, ];
        X = sdf[, 6:ncol(sdf)];
        y = sdf[, 5];


        print_info = paste(class, instance, nrow(X), ncol(X), nrow(y), ncol(y), sep=" ");
        print(print_info)

        features = calculate_features(X, y);

        features <- c(features, instance=instance);
        features <- c(features, class=class);

        #datalist <- append(datalist, features);

        datalist[[class*numinstances+instance]] <- features;
      }

      big_data = do.call(rbind, datalist);
      print(big_data);
    }


    write.csv(big_data, write_file, row.names = FALSE);
}



