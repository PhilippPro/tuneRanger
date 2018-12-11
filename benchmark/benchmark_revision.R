library(checkpoint)
dir = paste0(getwd(), "/benchmark")
checkpoint("2018-03-03", project = dir)

library(devtools)
library(OpenML)
load_all("../tuneRanger")

# Compare runtime and AUC/Brier Score with mlr
library(mlr)

library(mlrHyperopt)
source("./benchmark/RLearner_classif_hyperoptRanger.R")

multiclass.au1p.control = makeHyperControl(
  mlr.control = makeTuneControlRandom(maxit = 25),
  resampling = cv10,
  measures = multiclass.au1p
)

brier.control = makeHyperControl(
  mlr.control = makeTuneControlRandom(maxit = 25),
  resampling = cv10,
  measures = multiclass.brier
)

logloss.control = makeHyperControl(
  mlr.control = makeTuneControlRandom(maxit = 25),
  resampling = cv10,
  measures = logloss
)

lrns = list(
  makeLearner("classif.hyperoptRanger", id = "hyperopt.AUC", predict.type = "prob", hyper.control = multiclass.au1p.control),
  makeLearner("classif.hyperoptRanger", id = "hyperopt.Brier", predict.type = "prob", hyper.control = brier.control),
  makeLearner("classif.hyperoptRanger", id = "hyperopt.Logloss", predict.type = "prob", hyper.control = logloss.control)
)

rdesc = makeResampleDesc("Holdout")
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
configureMlr(on.learner.error = "warn")
set.seed(126)
bmr_rev1 = benchmark(lrns, iris.task, rdesc, measures)

library(OpenML)
load("./benchmark/data/time.estimate.RData")
rdesc = makeResampleDesc("RepCV", reps = 10, folds = 5)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
# benchmark
bmr_rev = list()
configureMlr(on.learner.error = "warn")

# Choose small and big datasets
# select datasets where RF do not take longer than ...

# take only small ones first; afterwards some bigger datasets
task.ids.bmr_rev = task.ids[which((unlist(time.estimate)-100)<60)]
cbind(time.estimate, (unlist(time.estimate)-100)<60)
unlist(time.estimate)[which((unlist(time.estimate)-100)<60)]
tasks[which((unlist(time.estimate)-100)<60),]

for(i in seq_along(task.ids.bmr_rev)) { # 13 datasets
  print(i)
  set.seed(200 + i)
  task = getOMLTask(task.ids.bmr_rev[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_rev[[i]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_rev, file = "./benchmark/data/bmr_rev.RData")
}
load("./benchmark/data/bmr_rev.RData")
# Which datasets are not super easy (AUC < 0.99) and discriminate between the algorithms?

# medium datasets (between 160 seconds and 10 minutes)
task.ids.bmr_rev2 = task.ids[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600)]
unlist(time.estimate)[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600 )]

# 13 datasets
for(i in seq_along(task.ids.bmr_rev2)) {
  print(i)
  set.seed(300 + i)
  task = getOMLTask(task.ids.bmr_rev2[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_rev[[length(bmr_rev) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_rev, file = "./benchmark/data/bmr_rev.RData")
}
load("./benchmark/data/bmr_rev.RData")

# big datasets (between 10 minutes and 1 hour)
task.ids.bmr_rev3 = task.ids[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
unlist(time.estimate)[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
# 9 datasets

rdesc = makeResampleDesc("CV", iters = 5)
bmr_rev_big = list()
# Hier evtl. doch ein paar Wiederholungen einbauen, da die Streuung sonst zu groß ist. 
# Zunächst einfach mal durchlaufen lassen (kann dannach hinzugefügt werden).
for(i in seq_along(task.ids.bmr_rev3)) {
  print(i)
  set.seed(400 + i) 
  task = getOMLTask(task.ids.bmr_rev3[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_rev_big[[length(bmr_rev_big) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_rev_big, file = "./benchmark/data/bmr_rev_big.RData")
}
load("./benchmark/data/bmr_rev_big.RData")

# Very big datasets, 4 datasets
tasks4 = tasks[which((unlist(time.estimate))>=3600),]
task.ids.bmr_rev4 = task.ids[which((unlist(time.estimate))>=3600)]
unlist(time.estimate)[which((unlist(time.estimate))>=3600)]

for(i in seq_along(task.ids.bmr_rev4)) {
  print(i)
  set.seed(400 + i) 
  task = getOMLTask(task.ids.bmr_rev4[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_rev_big[[length(bmr_rev_big) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_rev_big, file = "./benchmark/data/bmr_rev_big.RData")
}
load("./benchmark/data/bmr_rev_big.RData")


