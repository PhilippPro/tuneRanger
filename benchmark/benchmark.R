library(devtools)
load_all("../tuneRF")

# Compare runtime and AUC/Brier Score with mlr
library(mlr)

source("./benchmark/RLearner_classif_caretRanger.R")
library(mlrHyperopt)
source("./benchmark/RLearner_classif_hyperoptRanger.R")

lrns = list(
  makeLearner("classif.tuneRF", id = "tuneRFAUC", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(multiclass.au1p))), 
  makeLearner("classif.tuneRF", id = "tuneRFBrier", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(multiclass.brier))), 
  makeLearner("classif.tuneRF", id = "tuneRFLogloss", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(logloss))), 
  makeLearner("classif.hyperoptRanger", id = "hyperopt", predict.type = "prob"), 
  makeLearner("classif.caretRanger", id = "caret", predict.type = "prob"), 
  makeLearner("classif.ranger", id = "ranger", par.vals = list(num.trees = 2000, respect.unordered.factors = TRUE), predict.type = "prob")
)

rdesc = makeResampleDesc("CV", iters = 2)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
configureMlr(on.learner.error = "warn")
set.seed(123)
bmr1 = benchmark(lrns, iris.task, rdesc, measures)

library(OpenML)
tasks = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, 
  data.tag = "study_14", estimation.procedure = "10-fold Crossvalidation")
task.ids = tasks$task.id
task.ids = task.ids[-49] # not a classification task

# time estimation
time.estimate = list()
for(i in seq_along(task.ids)) {
  print(i)
  task = getOMLTask(task.ids[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  time.estimate[[i]] = estimateTuneRFTime(task, num.threads = 10, num.trees = 1000)
  print(time.estimate[[i]])
  save(time.estimate, file = "./benchmark/time.estimate.RData")
}

rdesc = makeResampleDesc("RepCV", reps = 5, folds = 5)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
# benchmark
bmr = list()
configureMlr(on.learner.error = "warn")


# Choose 5 (10) small, 5 medium and 5 big datasets
# maybe take jakobs wrapper

#library(batchtools)
#tmp = makeExperimentRegistry(file.dir = "./benchmark/batch", make.default = FALSE)
# select randomly 20 datasets that do not take longer than 1 hour and benchmark the algorithms
set.seed(127)
# task.ids.bmr = task.ids[sample(which(time.estimate<3600), 20)]
# take only small ones first; afterwards some bigger datasets
task.ids.bmr = task.ids[which((unlist(time.estimate)-100)*2<100)]
cbind(time.estimate, (unlist(time.estimate)-100)*2<600)

for(i in seq_along(task.ids.bmr)) {
  print(i)
  set.seed(145 + i)
  task = getOMLTask(task.ids.bmr[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[i]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}

# Which datasets are not super easy (AUC < 0.99)and discriminate between the algorithms?


# 80 h auf einem Core -> 8 h
bmr
