library(SpiecEasi)
ff = 1
n = 100
d = 100*ff
e = d
p1 = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
p2 = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
p3 = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
mapping = c("0p0", "0p1", "0p2", "0p3", "0p4", "0p5", "0p6", "0p7", "0p8", "0p9", "1p0")
type = "type4"
if(type == "type1"){
  p1 = c(0.5)
  p2 = c(1.0)
  p3 = c(1.0)
}
if(type == "type2"){
  p1 = c(1.0)
  p2 = c(0.5)
  p3 = c(0.0)
}
if(type == "type3"){
  p1 = c(1.0)
  p2 = c(0.5)
  p3 = c(1.0)
}
if(type == "type4"){
  p1 = c(1.0)
  p2 = c(1.0)
  p3 = c(1.0)
}
n_class = 8;
configure = 2;
if (n_class == 2){
  if(configure == 1){
    k_list <- list(c(20*ff, 10*ff, 70*ff), c(30*ff, 20*ff, 50*ff))
  }
  if(configure == 2){
    k_list <- list(c(10*ff, 20*ff, 70*ff), c(10*ff, 20*ff, 70*ff))
  }
}
if(n_class == 4){
  if(configure == 1){
    k_list <- list(c(10*ff, 20*ff, 70*ff), c(30*ff, 20*ff, 50*ff), c(20*ff, 30*ff, 50*ff), c(40*ff, 10*ff, 50*ff));
  }
  if(configure == 2){
    k_list <- list(c(10*ff, 20*ff, 70*ff), c(10*ff, 20*ff, 70*ff), c(10*ff, 20*ff, 70*ff), c(10*ff, 20*ff, 70*ff));
  }
}
if(n_class==8){
  if(configure == 1){
    k_list <- list(c(10*ff, 20*ff, 70*ff), c(30*ff, 20*ff, 50*ff), 
                   c(20*ff, 30*ff, 50*ff), c(40*ff, 10*ff, 50*ff), 
                   c(50*ff, 20*ff, 30*ff), c(30*ff, 10*ff, 60*ff), 
                   c(20*ff, 50*ff, 30*ff), c(40*ff, 50*ff, 10*ff));
  }
  if(configure == 2){
    k_list <- list(c(10*ff, 20*ff, 70*ff), c(10*ff, 20*ff, 70*ff), 
                   c(10*ff, 20*ff, 70*ff), c(10*ff, 20*ff, 70*ff), 
                   c(10*ff, 20*ff, 70*ff), c(10*ff, 20*ff, 70*ff), 
                   c(10*ff, 20*ff, 70*ff), c(10*ff, 20*ff, 70*ff));
  }
}


lower <- c(5, 1, 0);
upper <- c(10, 5, 1);

# for(class in 1:n_class){
#   index <- c(1:d);
#   index <- sample(index);
#   #print(index)
#   mu = 0
#   k = k_list[[class]]
#   for(i in 1:length(k)){
#     mu  = c(mu, runif(k[i],lower[i], upper[i]));
#   }
#   mu = mu[-1]
#   mu = mu[index]
#   Cor <- diag(d)
#   new_data <- rmvzinegbin(n, mu=mu, Sigma=Cor);
#   if(class == 1){
#     data <- new_data
#     label <- matrix(class-1, n, 1);
#   }
#   else{
#     data <- rbind(data, new_data)
#     label <- rbind(label, matrix(class-1, n, 1))
#   }
# }
# org_data = data;
# histData = hist(org_data, breaks = 25)
# histData$counts = histData$counts/sum(histData$counts)
# plot(histData, ylab='fraction')



for(i1 in 1:length(p1)){
  for(i2 in 1:length(p2)){
    for(i3 in 1:length(p3)){
      for (class in 1:n_class){
        new_data <- org_data[( (class-1)*n+1):(class*n),1:d]
        count = 0
        for (i in 1:(n*d) ){
          if(new_data[i] == 0){
            if(runif(1,0, 1) > p1[i1]){
              new_data[i] = new_data[i] + sample.int(5, 1, replace = T )
              count = count + 1
            }
          }
          else{
            if(runif(1, 0, 1) > p2[i2]){
              if(runif(1, 0, 1) > p3[i3]){
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
    }
  }
}
histData = hist(data, breaks = 25)
#histData$counts = log(histData$counts+1)
histData$counts = histData$counts/sum(histData$counts)
plot(histData, ylab=NULL, xlab = NULL, main = NULL, xlim = c(0, 10), ylim = c(0,1))
