library(SpiecEasi)

total_train_sample = as.matrix(read.table("/Users/chiehlo/Desktop/HMP_project/deep/datasets/IBDSet/Data/IBD/Splits/IT4/TrainSample_4.txt"))
total_train_label = as.matrix(read.table("/Users/chiehlo/Desktop/HMP_project/deep/datasets/IBDSet/Data/IBD/Splits/IT4/TrainLabel_4.txt"))

class0 <- total_train_sample[total_train_label[,1] == 1, ];
class1 <- total_train_sample[total_train_label[,2] == 1, ];


percentage <- c(0.5, 0.75, 1.0, 1.5, 2.0)
for (i in 1:length(percentage)){
  print(i)
  n_class0 <- round(nrow(class0)*percentage[i])
  n_class1 <- round(nrow(class1)*percentage[i])

  X0 <- synth_comm_from_counts(class0, mar=2, distr='negbin',  n=n_class0 )
  X1 <- synth_comm_from_counts(class1, mar=2, distr='negbin',  n=n_class1 )
  X <- rbind(X0, X1);
  label0 <- matrix(0, n_class0, 1);
  label1 <- matrix(1, n_class1, 1);
  labels <- rbind(label0, label1);

  sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/IBDSet/Data/IBD/Augmentation/IT4/nbSample", "_4_", round(percentage[i]*100), ".txt", sep = "");
  label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/IBDSet/Data/IBD/Augmentation/IT4/nbLabel", "_4_", round(percentage[i]*100), ".txt", sep = "");
  write.table(X, sample_file, sep="\t", row.names = F, col.names = F)
  write.table(labels, label_file, sep="\t", row.names = F, col.names = F)

}



add_on <- c(500, 1000, 2000, 3000)

for (i in 1:length(add_on)){
  print(i)
  n_class0 <- add_on[i]
  n_class1 <- add_on[i]
  X0 <- synth_comm_from_counts(class0, mar=2, distr='negbin',  n=n_class0 )
  X1 <- synth_comm_from_counts(class1, mar=2, distr='negbin',  n=n_class1 )
  X <- rbind(X0, X1);
  label0 <- matrix(0, n_class0, 1);
  label1 <- matrix(1, n_class1, 1);
  labels <- rbind(label0, label1);

  sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/IBDSet/Data/IBD/Augmentation/IT4/nbSample", "_4_", add_on[i], ".txt", sep = "");
  label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/IBDSet/Data/IBD/Augmentation/IT4/nbLabel", "_4_", add_on[i], ".txt", sep = "");
  write.table(X, sample_file, sep="\t", row.names = F, col.names = F)
  write.table(labels, label_file, sep="\t", row.names = F, col.names = F)
}


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




