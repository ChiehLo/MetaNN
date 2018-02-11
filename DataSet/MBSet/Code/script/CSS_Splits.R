OTU <- read.table("/Users/chiehlo/Desktop/HMP_project/deep/MBenchmark/benchmarks/CSS/CSS_otus.txt")
Labels <- read.table("/Users/chiehlo/Desktop/HMP_project/deep/MBenchmark/benchmarks/CSS/CSS_labels.txt", sep = '\t')
test_indices <- read.table("/Users/chiehlo/Desktop/HMP_project/deep/MBenchmark/benchmarks/CSS/CSS_test_indices.txt", sep = '\t')
spS <- read.table("/Users/chiehlo/Desktop/HMP_project/deep/MBenchmark/spCSS.txt", sep = ' ')
nSamples <- nrow(OTU)
nOTU <- ncol(OTU)


sampleName <- strsplit(as.character(spS[,2]), ':')
spS <- as.matrix(spS[,3])
rownames(spS) <- sampleName
count_matrix <- matrix(0, nSamples, nOTU)
for (i in 1:nSamples){
  index <- which(rownames(spS) == as.character(rownames(OTU)[i]))
  count_matrix[i,] <- round(spS[index]*as.matrix(OTU[i,]) )
}
rownames(count_matrix) <- rownames(OTU)

##### OTU filtering
OTU_occur <- matrix(0, ncol(OTU),1)
for (i in 1:ncol(OTU)){
  OTU_occur[i] <- sum(OTU[,i]!=0)
}

a <- sum(OTU_occur > 0.03*nrow(OTU))
index <- OTU_occur > 12
count_matrix_filter <- count_matrix[,index]; 

Total_split <- 1
for (IT in 1:Total_split){
  #### train-test split
  train <- count_matrix_filter[test_indices[,IT] == "FALSE", ]
  test <- count_matrix_filter[test_indices[,IT] == "TRUE", ]
  train_label <- as.matrix(as.numeric(Labels[ test_indices[,IT] == "FALSE" , 2 ]))
  test_label <- as.matrix(as.numeric(Labels[ test_indices[,IT] == "TRUE" , 2 ])) 
  train_label = train_label - 1;
  test_label = test_label -1;
  
  train_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/CSS/Splits/IT", IT, "/TrainSample_",  IT , ".txt", sep = "");
  train_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/CSS/Splits/IT", IT, "/TrainLabel_", IT, ".txt", sep = "");
  test_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/CSS/Splits/IT", IT, "/TestSample_",  IT , ".txt", sep = "");
  test_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/datasets/MBSet/Data/CSS/Splits/IT", IT, "/TestLabel_", IT, ".txt", sep = "");
  
  write.table(train, train_sample_file, sep="\t", row.names = F, col.names = F)
  write.table(train_label, train_label_file, sep="\t", row.names = F, col.names = F)  
  write.table(test, test_sample_file, sep="\t", row.names = F, col.names = F)
  write.table(test_label, test_label_file, sep="\t", row.names = F, col.names = F)  
}




