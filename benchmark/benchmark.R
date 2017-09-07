library(devtools)
load_all("../tuneRF")

# Compare runtime and AUC/Brier Score with mlr
library(mlr)

source("./benchmark/RLearner_classif_caretRanger.R")
source("./benchmark/RLearner_classif_hyperoptRanger.R")

lrns = list(makeLearner("classif.tuneRF", id = "tuneRFBrier", predict.type = "prob", par.vals = list(num.trees = 2000, measure = list(multiclass.brier))), 
  makeLearner("classif.tuneRF", id = "tuneRFAUC", predict.type = "prob", par.vals = list(num.trees = 2000, measure = list(multiclass.au1p))), 
  makeLearner("classif.hyperoptRanger", id = "hyperopt", predict.type = "prob"), 
  makeLearner("classif.caretRanger", id = "caret", predict.type = "prob"), 
  makeLearner("classif.ranger", id = "ranger", par.vals = list(num.trees = 2000, respect.unordered.factors = TRUE), predict.type = "prob")
)

bmr1 = benchmark(lrns[[3]], iris.task)

library(OpenML)
tasks = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, 
  data.tag = "study_14", estimation.procedure = "10-fold Crossvalidation")
task.ids = tasks$task.id

rdesc = makeResampleDesc("RepCV", reps = 2, folds = 5)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)

# time estimation
time.estimate = list()
for(i in seq_along(task.ids)) {
  print(i)
  task = getOMLTask(task.ids[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  time.estimate[[i]] = estimateTuneRFTime(task)
  print(time.estimate[[i]])
}
save(time.estimate, file = "./benchmark/time.estimate.RData")
# benchmark
bmr = list()
configureMlr(on.learner.error = "warn")
for(i in seq_along(task.ids)) {
  set.seed(145 + i)
  if(time.estimate[[i]] < 300){
    task = getOMLTask(task.ids[i])
    task = convertOMLTaskToMlr(task)$mlr.task
    bmr[[i]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  }
  save(bmr, file = "./benchmark/bmr.RData")
}

