getwd()
setwd("/home/jmmoon/문서/wsvm")
source("wsvm.r")
library(e1071)
library(caret)

set.seed(1)
data(iris)
iris.binary <- iris[iris$Species != "setosa",]#only use two classes

#partition into training and test dataset
idx.training <- createDataPartition(iris.binary $Species, p = .75, list = FALSE)
training <- iris.binary [ idx.training,]
testing  <- iris.binary [-idx.training,]

#make type vector indicating whether the sample belongs to majority, minority or synthetic sample
label <- training[,ncol(training)]
index.full <- 1:length(label)
type <- 1:length(label)
index.syn <- sample(index.full, 30)
type[label == "virginica"] <- "maj"
type[label == "versicolor"] <- "min"
type[index.syn] <- "syn"

#turn label vector into -1 and 1 for suppor vector machine fitting
y.values <- -1 * (label == "versicolor") + 1 * (label == "virginica")
y <- as.factor(y.values)

#scale x
x <- training[,-ncol(training)]
training.scaler <- caret::preProcess(x, method = c("center", "scale"))
x.scaled <- predict(training.scaler, x)


#fit weighted rbf svm
fit.result <- wsvm.fit(x = x.scaled, y = y, type = type, three.weights = list(maj = 0.5, min = 0.7, syn = 0.3), kernel = list(type = "rbf", par = 1/4))
fit.result$alpha.sv #result: computed Lagrange multipliers
fit.result$bias #result: computed bias term of the hyperplane

#evaluate the fitted model with the test set
testing.X <- testing[,-ncol(testing)]
testing.Y <- testing[,ncol(testing)]
testing.Y <- as.factor(-1 * (testing.Y == "versicolor") + 1 * (testing.Y == "virginica"))

testing.X <- predict(training.scaler , testing.X)

pred <-wsvm.predict(testing.X, fit.result)$predicted.Y
confusionMatrix(pred, testing.Y) #confusion matrix
