library(devtools)
load_all("../tuneRF")

# Compare runtime and AUC/Brier Score with mlr
library(mlr)

source("./benchmark/RLearner_classif_caretRanger.R")
library(mlrHyperopt)
source("./benchmark/RLearner_classif_hyperoptRanger.R")

lrns = list(
  makeLearner("classif.tuneRF", id = "tuneRFMMCE", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(mmce))),
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
task.ids = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, data.tag = "OpenML100", estimation.procedure = "10-fold Crossvalidation")$task.id
task.ids = task.ids[-47]
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
load("./benchmark/time.estimate.RData")
rdesc = makeResampleDesc("RepCV", reps = 10, folds = 5)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
# benchmark
bmr = list()
configureMlr(on.learner.error = "warn")

# Choose 5 (10) small, 5 medium and 5 big datasets
# select datasets where RF do not take longer than ...
set.seed(127)
# task.ids.bmr = task.ids[sample(which(time.estimate<3600), 20)]
# take only small ones first; afterwards some bigger datasets
task.ids.bmr = task.ids[which((unlist(time.estimate)-100)<60)]
cbind(time.estimate, (unlist(time.estimate)-100)<60)

namen = numeric(30)
for(i in seq_along(task.ids.bmr)) {
  print(i)
  task = getOMLTask(task.ids.bmr[i])
  namen[i] = task$input$data.set$desc$name
}

# each dataset only once
task.ids.bmr = task.ids.bmr[-c(19, 20, 22:30)]

for(i in seq_along(task.ids.bmr)) {
  print(i)
  set.seed(145 + i)
  task = getOMLTask(task.ids.bmr[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[i]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}
load("./benchmark/bmr.RData")
# Which datasets are not super easy (AUC < 0.99) and discriminate between the algorithms?

# 80 h auf einem Core -> 8 h

# medium datasets (between 160 seconds and 10 minutes)
task.ids.bmr2 = task.ids[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600)]
unlist(time.estimate)[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600 )]
namen = numeric(length(task.ids.bmr2))
for(i in seq_along(task.ids.bmr2)) {
  print(i)
  task = getOMLTask(task.ids.bmr2[i])
  namen[i] = task$input$data.set$desc$name
}
task.ids.bmr2 = task.ids.bmr2[-c(8, 13:18)]
# 11 datasets

for(i in seq_along(task.ids.bmr2)) {
  print(i)
  set.seed(145 + i)
  task = getOMLTask(task.ids.bmr2[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[length(bmr) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}
load("./benchmark/bmr.RData")

# Analysis

# Data cleaning
# caret problems
a = bmr[[30]]
a$results$Australian$caret$measures.test
# no results at all
# mmce
a = bmr[[20]]
resis = a$results$`kr-vs-kp`$tuneRFMMCE$measures.test
resis = resis[!is.na(resis$mmce),]
names = names(bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr)
bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr = colMeans(resis)[2:6]
names(bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr) = names

# Wilcoxon paired test
library(mlr)
data = array(NA, dim = c(length(bmr), 2, 5))
for(i in 1:length(bmr)) {
  print(i)
  res_aggr = data.frame(getBMRAggrPerformances(bmr[[i]]))
  data[i,,] = t(data.frame(res_aggr[,c(3,7) ]))
}

# mmce
wilcox.test(data[,1,1], data[,2,1], alternative = "less", paired = TRUE)
wilcox.test(data[,1,1], data[,2,1], paired = TRUE)
# auc
wilcox.test(data[,1,2], data[,2,2], alternative = "greater", paired = TRUE)
wilcox.test(data[,1,2], data[,2,2], paired = TRUE)
t.test(data[,1,2], data[,2,2], alternative = "greater", paired = TRUE)
t.test(data[,1,2], data[,2,2], paired = TRUE)
# brier
wilcox.test(data[,1,3], data[,2,3], alternative = "less", paired = TRUE)
wilcox.test(data[,1,3], data[,2,3], paired = TRUE)
# logloss
wilcox.test(data[,1,4], data[,2,4], alternative = "less", paired = TRUE)
wilcox.test(data[,1,4], data[,2,4], paired = TRUE)

mean(data[,1,1], na.rm = T)
mean(data[,2,1], na.rm = T)

# Descriptive Analysis
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr[[1]]))
res_aggr = resi[[1]]
res_aggr_rank = apply(resi[[1]], 1, rank)

for(i in 2:length(bmr)) {
  resi[[i]] = data.frame(getBMRAggrPerformances(bmr[[i]]))
  # caret gets the worst result, if NA
  if(is.na(resi[[i]][1,6])) {
    resi[[i]][1,6] = max(resi[[i]][1,], na.rm = T)
    resi[[i]][2,6] = min(resi[[i]][2,], na.rm = T)
    resi[[i]][3,6] = max(resi[[i]][3,], na.rm = T)
    resi[[i]][4,6] = max(resi[[i]][4,], na.rm = T)
  }
  res_aggr = res_aggr + resi[[i]]
  res_aggr_rank = res_aggr_rank + apply(resi[[i]], 1, rank)
}
res_aggr = res_aggr/length(bmr)
colnames(res_aggr) = sapply(lrns, getLearnerId)
t(res_aggr)

# average rank matrix
res_aggr_rank = res_aggr_rank/length(bmr)
rownames(res_aggr_rank) = sapply(lrns, getLearnerId)
res_aggr_rank

# Analysis of time
time = matrix(NA, length(bmr),  ncol(resi[[1]]))
for(i in 1:30) {
  time[i,] = unlist(resi[[i]][5,])
}
time_order = order(time[,1])
time = time[time_order,]
plot(time[,1], type = "l", ylim = c(0, max(time, na.rm = T)), ylab = "Time in seconds", xlab = "Dataset number")
for(i in 2:ncol(time))
  lines(time[,i], col = i)
legend("topleft", legend = sapply(lrns, getLearnerId), col = 1:ncol(time), lty = 1)

# Graphical analysis of performance
perfi = matrix(NA, length(bmr),  ncol(resi[[1]]))
for(j in c(1:4)) {
  for(i in 1:30) {
    perfi[i,] = unlist(resi[[i]][j,])
  }
  #perfi = perfi[time_order,]
  perfi = perfi[order(perfi[,1]),]
  print(plot(perfi[,1], type = "l", ylim = c(min(perfi, na.rm = T), max(perfi, na.rm = T)), ylab = rownames(res_aggr)[j], xlab = "Dataset number"))
  for(i in 2:ncol(time))
    lines(perfi[,i], col = i)
  legend("topleft", legend = sapply(lrns, getLearnerId), col = 1:ncol(perfi), lty = 1)
}


# Anhang
# regression

# big (between 10 minutes and 1 hour)
task.ids.bmr3 = task.ids[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
namen = numeric(length(task.ids.bmr3))
for(i in seq_along(task.ids.bmr3)) {
  print(i)
  task = getOMLTask(task.ids.bmr3[i])
  namen[i] = task$input$data.set$desc$name
}
task.ids.bmr3 = task.ids.bmr3[-c(8, 10:13)]
# 8 datasets

rdesc = makeResampleDesc("RepCV", reps = 2, folds = 5)
for(i in seq_along(task.ids.bmr3)) {
  print(i)
  set.seed(245 + i)
  task = getOMLTask(task.ids.bmr3[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[length(bmr) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}
load("./benchmark/bmr.RData")


# very big (more than one hour (up to 4 hours))
task.ids.bmr4 = task.ids[which((unlist(time.estimate))>3600)]
unlist(time.estimate)[which((unlist(time.estimate))>3600)]
namen = numeric(length(task.ids.bmr4))
for(i in seq_along(task.ids.bmr4)) {
  print(i)
  set.seed(345 + i)
  task = getOMLTask(task.ids.bmr4[i])
  namen[i] = task$input$data.set$desc$name
}
task.ids.bmr4 = task.ids.bmr4[-c(5:9)]
# 4 datasets

rdesc = makeResampleDesc("RepCV", reps = 1, folds = 5)
for(i in seq_along(task.ids.bmr4)) {
  print(i)
  set.seed(145 + i)
  task = getOMLTask(task.ids.bmr4[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[length(bmr) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}
load("./benchmark/bmr.RData")

# save task.id vectors...
# small datasets have very volatile estimation of the error rate/AUC; that's why tuneRF is sometimes worse.
