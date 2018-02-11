# # parameter settings
# n = 200    
# n.dam = 20   
# b0 = c(0.4, 0.55)
# cor = c(0.5, 0.8)
# 
# # data simulation
# corr = runif(1, cor[1], cor[2])
# x = sim.x(n = n, m = 20, corr = corr)
# q = rep(1/n.dam, n.dam-1)
# q = cumsum(q)
# quantiles = quantile(x[,1], q)
# dam = as.numeric( factor(cut(x[,1], breaks = c(-Inf, quantiles, Inf))) )   
# quantiles = quantile(x[,2], 0.45)
# diet = as.numeric( factor(cut(x[,2], breaks = c(-Inf, quantiles, Inf))) )
# diet = diet - 1   
# 
# da = rep(NA, n.dam)
# sigma = runif(1, 0.5, 1)
# for (j in 1:n.dam) da[j] = rnorm(1, 0, sigma)
# mu0 = runif(n, 0.1, 3.5)
# theta = runif(1, 0.1, 5) 
# b = runif(1, b0[1], b0[2]) 
# ys = sim.y(x = diet, mu = mu0 + da[dam], sigma = 1, coefs = b, p.neg = 0, nb.theta = theta) 
# y0 = ys$y.nb
# N = exp(mu0)

ff = 5
n = 100
d = 100*ff
e = d
n_class = 2;
k_list <- list(c(10*ff, 20*ff, 70*ff), c(30*ff, 20*ff, 50*ff));
#k_list <- list(c(10, 20, 70), c(10, 20, 70));
lower <- c(5, 1, 0);
upper <- c(10, 5, 1);
for (class in 1:n_class){
  index <- c(1:d);
  index <- sample(index);
  #print(index)
  mu = 0
  k = k_list[[class]]
  for(i in 1:length(k)){
    mu  = c(mu, runif(k[i],lower[i], upper[i]));
  }
  mu = mu[-1]
  mu = mu[index]
  #print(sum(mu>5))
  #set.seed(10010)
  # graph <- make_graph('scale_free', d, e)
  # Prec  <- graph2prec(graph)
  # Cor   <- cov2cor(prec2cov(Prec))
  Cor <- diag(d)
  drop <- c(1:(n*d));
  drop <- sample(drop);
  new_data <- rmvzinegbin(n, mu=mu, Sigma=Cor);
  #new_data[drop[1:round(0.1*(n*d))]] <- 0;
  is = 0.1
  #new_data[drop[(round(0.0*n*d)+1):round(is*n*d)]] <- sample.int(100, (round(is*n*d)-round(0.0*n*d)), replace = T )
  #new_data[drop] <- sample.int(20, n*d, replace = T )
  count = 0
  for (i in 1:(n*d) ){
    if(new_data[i] == 0){
      if(runif(1,0, 1) > 0.5){
        new_data[i] = new_data[i] + sample.int(5, 1, replace = T )
        count = count + 1
      }
    }
    else{
      if(runif(1,0, 1) > 0.2){
        if(runif(1,0,1) > 0.5){
          #new_data[i] = new_data[i] + sample.int(5, 1, replace = T )
          temp = round(1*new_data[i]) + 1;
          new_data[i] = new_data[i] + sign(runif(1,-1,1))*sample.int(temp, 1, replace = T )
          if(new_data[i] < 0){
            new_data[i] = 0
          }
          count = count + 1 
        }
        else{
          new_data[i]  = 0
          count = count + 1 
        }
      }
    }
  }
  print(count)
  if(class == 1){
    data <- new_data
    label <- matrix(class-1, n, 1);
  }
  else{
    data <- rbind(data, new_data)
    label <- rbind(label, matrix(class-1, n, 1))
  }
}
ratio = 0.67;
for (IT in 1:10){
  x_d <- 1:n
  x_n <- (n+1):(2*n)
  set.seed(IT);
  split_d <- sample(x_d);
  set.seed(IT);
  split_n <- sample(x_n);
  
  train_d <-  data[split_d[1:round(n*ratio)], ]
  train_n <-  data[split_n[1:round(n*ratio)], ]
  test_d <-   data[split_d[(round(n*ratio)+1):n], ]
  test_n <-   data[split_n[(round(n*ratio)+1):n], ]
  train_label <- rbind(matrix(0, nrow(train_d), 1), matrix(1, nrow(train_n), 1) )
  test_label <- rbind(matrix(0, nrow(test_d), 1), matrix(1, nrow(test_n), 1) )
  Synthetic_train <- rbind(train_d, train_n);
  Synthetic_test <- rbind(test_d, test_n);
  train_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/Synthetic/IT", IT, "/TrainSampleSynthetic_",  IT , ".txt", sep = "");
  train_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/Synthetic/IT", IT, "/TrainLabelSynthetic_", IT, ".txt", sep = "");
  test_sample_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/Synthetic/IT", IT, "/TestSampleSynthetic_",  IT , ".txt", sep = "");
  test_label_file <- paste("/Users/chiehlo/Desktop/HMP_project/deep/Synthetic/IT", IT, "/TestLabelSynthetic_", IT, ".txt", sep = "");
  
  write.table(Synthetic_train, train_sample_file, sep="\t", row.names = F, col.names = F)
  write.table(train_label, train_label_file, sep="\t", row.names = F, col.names = F)
  write.table(Synthetic_test, test_sample_file, sep="\t", row.names = F, col.names = F)
  write.table(test_label, test_label_file, sep="\t", row.names = F, col.names = F)
  
}


# index <- c(1:d);
# index <- sample(index);
# k <- c(100, 300, 100);
# lower <- c(5, 1, 0);
# upper <- c(10, 5, 1);
# mu = 0
# count = 1
# for(i in 1:length(k)){
#   mu  = c(mu, runif(k[i],lower[i], upper[i]));
# }
# mu = mu[-1]
# mu = mu[index]
# 
# set.seed(10010)
# graph <- make_graph('scale_free', d, e)
# Prec  <- graph2prec(graph)
# Cor   <- cov2cor(prec2cov(Prec))
# data <- rmvzinegbin(n, mu=mu, Sigma=Cor)
#a <- cor(data)
# #rmvnegbin
# #rmvzinegbin