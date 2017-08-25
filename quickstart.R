library(devtools)
load_all("../tuneRF")
# roxygen2::roxygenise("../tuneRF")
# install("../tuneRF", dependencies = character(0))

# iris is a bit nonsense here
estimateTuneRFTime(iris.task)
set.seed(123)
res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, num.threads = 2, iters = 100)

res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, num.threads = 2, iters = 100, 
  parameters = list(replace = FALSE), tune.parameters = c("mtry", "sample.fraction", "respect.unordered.factors"))

# Best 5 % of the results
results = res$results
results[results$multiclass.brier >= quantile(results$multiclass.brier, 0.95),]

# Restart after failing in one of the iterations:
res = restartTuneRF("./optpath.RData", iris.task, measure = list(multiclass.brier))
