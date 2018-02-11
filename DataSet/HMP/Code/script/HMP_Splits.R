bodysite = c('Anterior_nares', 'Buccal_mucosa', 'Stool', 'Supragingival_plaque', 'Tongue_dorsum')

for (i in 1:length(bodysite)){
  bodySite = bodysite[i];
  pre_path = "/Users/chiehlo/Desktop/HMP_project/data/HMQCP/v35/";
  count_path = paste(pre_path, bodySite, "/genus_count_full.csv", sep = '');
  rawData <- read.csv(count_path, sep = "\t", header=FALSE)
  if(i == 1){
    full_data <- rawData;
    labels <- matrix( (i-1), ncol(rawData),1)
  }
  else{
    full_data <- cbind(full_data, rawData);
    labels <- rbind(labels, matrix((i-1), ncol(rawData), 1))
  }
  #print(nrow(rawData))
  #print(ncol(rawData))
}
rowSS <- rowSums(full_data)
index = which(rowSS==0)
filter_data <- t(as.matrix(full_data[-index, ]))

n_class = 5;
ratio = 0.67
for (IT in 1:10){
  for (cc in 1:n_class){
    x_d <- which(labels==(cc-1))
    n = length(x_d)
    set.seed(IT);
    split_d <- sample(x_d);
    train_d <-  filter_data[split_d[1:round(n*ratio)], ]
    test_d <-   filter_data[split_d[(round(n*ratio)+1):n], ]
    if (cc==1){
      Synthetic_train <- train_d
      Synthetic_test <- test_d
      train_label <- matrix((cc-1), nrow(train_d), 1)
      test_label <- matrix((cc-1), nrow(test_d), 1)
    }
    else{
      Synthetic_train <- rbind(Synthetic_train, train_d);
      Synthetic_test <- rbind(Synthetic_test, test_d);
      train_label <- rbind(train_label, matrix( (cc-1), nrow(train_d), 1))
      test_label <- rbind(test_label, matrix( (cc-1), nrow(test_d), 1))
    }
  }
  train_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/HMP/Data/Splits/IT", IT, "/TrainSampleHMP_",  IT , ".txt", sep = "");
  train_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/HMP/Data/Splits/IT", IT, "/TrainLabelHMP_", IT, ".txt", sep = "");
  test_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/HMP/Data/Splits/IT", IT, "/TestSampleHMP_",  IT, ".txt", sep = "");
  test_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/HMP/Data/Splits/IT", IT, "/TestLabelHMP_", IT, ".txt", sep = "");
  
  write.table(Synthetic_train, train_sample_file, sep="\t", row.names = F, col.names = F)
  write.table(train_label, train_label_file, sep="\t", row.names = F, col.names = F)
  write.table(Synthetic_test, test_sample_file, sep="\t", row.names = F, col.names = F)
  write.table(test_label, test_label_file, sep="\t", row.names = F, col.names = F)
  
}