library(devtools)
load_all("tuneRF")
# install("tuneRF")

# iris is a bit nonsense here
unlink("./optpath.RData")
res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, num.threads = 8, iters = 100)

res[res$y < quantile(res$y, 0.05),]

formula = as.formula(c("Species ~ ."))
estimateTuneRangerTime(formula, data = iris)
unlink("./optpath.RData")
res = tuneRanger(formula, data = iris, measure = measureMSE, num.trees = 1000, num.threads = 8, iters = 100)

