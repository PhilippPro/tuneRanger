library(caret)
train(iris[, 1:4], iris[, 5], method = "ranger")

library(mlrHyperopt)
hyperopt(iris.task, learner = "classif.ranger")

library(tuneRF)
tuneRF(iris.task)

# Compare runtime and AUC/Brier Score

