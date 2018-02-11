library(SpiecEasi)
datasetNames <- c('CSS', 'CS', 'CBH', 'FS', 'FSH')
dataset <- datasetNames[1];
Total_split <- 6
for (IT in 6:Total_split){
  train_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/", dataset, "/Splits/IT", IT, "/TrainSample_",  IT , ".txt", sep = "");
  train_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/", dataset, "/Splits/IT", IT, "/TrainLabel_", IT, ".txt", sep = "");
  total_train_sample = as.matrix(read.table(train_sample_file))
  total_train_label = as.matrix(read.table(train_label_file))
  class_samples <- list()
  nClasses <- nrow(unique(total_train_label))
  for (i in 1:(nClasses) ){
    class_samples[[i]] <- total_train_sample[total_train_label == (i-1), ];
  }
  percentage <- c(0.5, 0.75, 1.0, 1.5, 2.0)
  percentage <- c(0.5)
  print(IT)
  for (i in 1:length(percentage)){
    print(i)
    x0 <- matrix(0, 1, ncol(class_samples[[1]]));
    labels <- matrix(0, 1, 1);
    for (j in 1:nClasses){
      samples <- class_samples[[j]];
      n_sample <- round(nrow(samples)*percentage[i])
      #n_sample <- percentage
      sigma <- var(samples)
      aa <- diag(sigma)!=0
      diag(sigma) = diag(sigma) + min(diag(sigma)[aa]);
      X_temp <- synth_comm_from_counts(samples, mar=2, distr='negbin',  n=n_sample, Sigma = sigma )
      X_temp[is.na(X_temp)] <- 0
      x0 <- rbind(x0, X_temp);
      label_temp <- matrix(j-1, nrow(X_temp), 1);
      labels <- rbind(labels, label_temp);
    }
    x0 <- x0[-1,];
    labels <- labels[-1,];
    nb_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/", dataset, "/Augmentation/IT", IT, "/nbSample_",  IT , "_", round(percentage[i]*100),  ".txt", sep = "");
    nb_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/", dataset, "/Augmentation/IT", IT, "/nbLabel_", IT, "_", round(percentage[i]*100), ".txt", sep = "");
    write.table(x0, nb_sample_file, sep="\t", row.names = F, col.names = F)
    write.table(labels, nb_label_file, sep="\t", row.names = F, col.names = F)
  }
  # add_on <- c(50, 100, 200, 300)
  # for (i in 1:length(add_on)){
  #   print(i)
  #   x0 <- matrix(0, 1, ncol(class_samples[[1]]));
  #   labels <- matrix(0, 1, 1);
  #   for (j in 1:nClasses){
  #     samples <- class_samples[[j]];
  #     n_sample <- add_on[i]
  #     sigma <- var(samples)
  #     aa <- diag(sigma)!=0
  #     diag(sigma) = diag(sigma) + min(diag(sigma)[aa]);
  #     X_temp <- synth_comm_from_counts(samples, mar=2, distr='negbin',  n=n_sample, Sigma = sigma )
  #     x0 <- rbind(x0, X_temp);
  #     label_temp <- matrix(j-1, nrow(X_temp), 1);
  #     labels <- rbind(labels, label_temp);
  #   }
  #   x0 <- x0[-1,];
  #   labels <- labels[-1,];
  #   nb_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/", dataset, "/Augmentation/IT", IT, "/nbSample_",  IT , "_", round(add_on[i]*nClasses),  ".txt", sep = "");
  #   nb_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/", dataset, "/Augmentation/IT", IT, "/nbLabel_", IT, "_", round(add_on[i]*nClasses), ".txt", sep = "");
  #   write.table(x0, nb_sample_file, sep="\t", row.names = F, col.names = F)
  #   write.table(labels, nb_label_file, sep="\t", row.names = F, col.names = F)
  # }
  
}











# add_on <- c(500, 1000, 2000, 3000)
# 
# for (i in 1:length(add_on)){
#   print(i)
#   n_class0 <- add_on[i]
#   n_class1 <- add_on[i]
#   X0 <- synth_comm_from_counts(class0, mar=2, distr='negbin',  n=n_class0 )
#   X1 <- synth_comm_from_counts(class1, mar=2, distr='negbin',  n=n_class1 )
#   X <- rbind(X0, X1);
#   label0 <- matrix(0, n_class0, 1);
#   label1 <- matrix(1, n_class1, 1);
#   labels <- rbind(label0, label1);
#   
#   sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/IBD_new_data/IT4/nbSample", "_4_", add_on[i], ".txt", sep = "");
#   label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/IBD_new_data/IT4/nbLabel", "_4_", add_on[i], ".txt", sep = "");
#   write.table(X, sample_file, sep="\t", row.names = F, col.names = F)
#   write.table(labels, label_file, sep="\t", row.names = F, col.names = F)
# }


# pp <- c(0.5, 0.75, 1, 1.5, 2)
# n_class0 <- round(nrow(class0)*pp);
# n_class1 <- round(nrow(class1)*pp);
# 
# add_on <- c(500, 1000, 2000, 3000)
# 
# n_class0 <- c(n_class0, add_on)
# n_class1 <- c(n_class1, add_on)



# X0 <- synth_comm_from_counts(class0, mar=2, distr='negbin',  n=sum(n_class0) )
# X1 <- synth_comm_from_counts(class1, mar=2, distr='negbin',  n=sum(n_class1) )
# X <- rbind(X0, X1);
# label0 <- matrix(0, n_class0, 1);
# label1 <- matrix(1, n_class1, 1);
# labels <- rbind(label0, label1);
# 
# write.table(X, "/Users/chiehlo/Desktop/HMP_project/deep/IBD_new_data/nbSample_1_500.txt", sep="\t", row.names = F, col.names = F)
# write.table(labels, "/Users/chiehlo/Desktop/HMP_project/deep/IBD_new_data/nbLabel_1_500.txt", sep="\t", row.names = F, col.names = F)




