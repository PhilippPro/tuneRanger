library(devtools)
load_all("../tuneRanger")
# roxygen2::roxygenise("../tuneRanger")
# install("../tuneRanger", dependencies = character(0))

# make an mlr task with the specific dataset (here iris)
# Classification task with makeClassifTask, Regression Task with makeRegrTask
iris.task = makeClassifTask(data = iris, target = "Species")

estimateTuneRangerTime(iris.task)
set.seed(123)
res = tuneRanger(iris.task, measure = list(multiclass.brier), num.trees = 1000, num.threads = 2, iters = 100, build.final.model = TRUE)

res = tuneRanger(iris.task, measure = list(multiclass.brier), num.trees = 1000, num.threads = 2, iters = 100, 
  parameters = list(replace = FALSE), tune.parameters = c("mtry", "sample.fraction", "respect.unordered.factors"))

# Best 5 % of the results
results = res$results
results[results$multiclass.brier < quantile(results$multiclass.brier, 0.05), ]

# Restart after failing in one of the iterations:
res = restartTuneRanger("./optpath.RData", iris.task, measure = list(multiclass.brier))
