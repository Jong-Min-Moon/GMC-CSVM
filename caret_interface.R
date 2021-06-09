getwd()
setwd("/home/jmmoon/문서/wsvm")
source("wsvm.r")
library(caret)

weighted.svm <- list(type = "Classification",
                             library = "kernlab",
                             prob = NULL,
                             loop = NULL) 



#1. paramters element
prm <- data.frame(parameter = c("sigma"),
                  class = c("numeric"),
                  label = c("Sigma"))
weighted.svm$parameters <- prm

#2. grid element
svmGrid <- function(x, y, len = NULL, search = "grid") {
  library(kernlab)
  ## This produces low, middle and high values for sigma 
  ## (i.e. a vector with 3 elements).
  x.values <- as.matrix( x[-dim(x)[2]])
  sigmas <- kernlab::sigest(x.values, na.action = na.omit, scaled = TRUE)  
  ## To use grid search:
  rng <- extendrange(log(sigmas), f = .75)
  out <- data.frame(sigma = exp(seq(from = rng[1], to = rng[2], length.out= len)))
  return(out)
}
weighted.svm$grid <- svmGrid

#3. fit element
svmFit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) { 
  wsvm.fit(x = x[-dim(x)[2]], y = y, type = x[dim(x)[2]], ..., kernel = list(type = "rbf", par = param$sigma)) 
}

weighted.svm$fit <- svmFit

#4. prediction
svmPred <- function(modelFit, newdata, preProc = NULL, submodels = NULL){
  pred <- wsvm.predict(newdata[-dim(newdata)[2]], modelFit)
  return(pred$predicted.Y)
}

weighted.svm$predict <- svmPred


#6. sort
svmSort <- function(x) x[order(x$sigma),]
weighted.svm$sort <- svmSort


set.seed(998)
data(iris)
iris.binary <- iris[iris$Species != "setosa",]
inTraining <- createDataPartition(iris.binary $Species, p = .75, list = FALSE)
training <- iris.binary [ inTraining,]
testing  <- iris.binary [-inTraining,]

label <- training[,5]
index.full <- 1:length(label)
type <- 1:length(label)

index.syn <- sample(index.full, 30)
type[label == "virginica"] <- "maj"
type[label == "versicolor"] <- "min"
type[index.syn] <- "syn"

y.values <- rep.int(1, length(label))
y.values[label != "virginica"] <- -1
y.values
y <- as.factor(y.values)
x <- cbind(training[,1:4], type)

fitControl <- trainControl(method = "repeatedcv",
                           ## 10-fold CV...
                           number = 5,
                           ## repeated ten times
                           repeats = 5)

set.seed(825)
Laplacian <- train(x, y,  
                   method = weighted.svm, 
                   preProc = c("center", "scale"),
                   tuneLength = 20,
                   trControl = fitControl,
                   three.weights = list(maj = 1, min = 1, syn = 1)
                   )
Laplacian
