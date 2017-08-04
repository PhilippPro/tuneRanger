library(devtools)
load_all("../tuneRF")
# roxygen2::roxygenise("../tuneRF")
# install("../tuneRF")

# iris is a bit nonsense here
unlink("./optpath.RData")
estimateTuneRFTime(iris.task)
res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, num.threads = 8, iters = 100)

res[res$multiclass.brier < quantile(res$multiclass.brier, 0.05),]





# Annex

# Alternatively use the ranger interface
# not recommended
# measure has to be defined with inputs "truth" and "response"
formula = as.formula(c("Species ~ ."))
estimateTuneRangerTime(formula, data = iris)
unlink("./optpath.RData")
res = tuneRanger(formula, data = iris, measure = measureMSE, num.trees = 1000, num.threads = 8, iters = 100)
