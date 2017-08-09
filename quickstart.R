library(devtools)
load_all("../tuneRF")
# roxygen2::roxygenise("../tuneRF")
# install("../tuneRF", dependencies = character(0))

# iris is a bit nonsense here
estimateTuneRFTime(iris.task)
res = tuneRF(iris.task, measure = list(multiclass.brier), num.trees = 1000, num.threads = 8, iters = 100)

# Best 5 % of the results
results = res$results
results[results$multiclass.brier < quantile(results$multiclass.brier, 0.05),]


# Restart after failing in one of the iterations:
res = restartTuneRF("./optpath.RData", iris.task, measure = list(multiclass.brier))
