library(devtools)
load_all("../tuneRanger")

# Compare runtime and AUC/Brier Score with mlr
library(mlr)

source("./benchmark/RLearner_classif_caretRanger.R")
library(mlrHyperopt)
source("./benchmark/RLearner_classif_hyperoptRanger.R")
library(randomForest)
source("./benchmark/RLearner_classif_tuneRF.R")

lrns = list(
  makeLearner("classif.tuneRanger", id = "tuneRFMMCE", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(mmce))),
  makeLearner("classif.tuneRanger", id = "tuneRFAUC", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(multiclass.au1p))),
  makeLearner("classif.tuneRanger", id = "tuneRFBrier", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(multiclass.brier))), 
  makeLearner("classif.tuneRanger", id = "tuneRFLogloss", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(logloss))), 
  makeLearner("classif.tuneRanger", id = "tuneRFMMCE_mtry", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(mmce), tune.parameters = "mtry")),
  makeLearner("classif.tuneRanger", id = "tuneRFAUC_mtry", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(multiclass.au1p), tune.parameters = "mtry")),
  makeLearner("classif.tuneRanger", id = "tuneRFBrier_mtry", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(multiclass.brier), tune.parameters = "mtry")),
  makeLearner("classif.tuneRanger", id = "tuneRFLogloss_mtry", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(logloss), tune.parameters = "mtry")),
  makeLearner("classif.hyperoptRanger", id = "hyperopt", predict.type = "prob"), 
  makeLearner("classif.caretRanger", id = "caret", predict.type = "prob"), 
  makeLearner("classif.tuneRF", id = "tuneRF", predict.type = "prob"), 
  makeLearner("classif.ranger", id = "ranger", par.vals = list(num.trees = 2000, num.threads = 10, respect.unordered.factors = "order"), predict.type = "prob")
)

rdesc = makeResampleDesc("CV", iters = 5)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
configureMlr(on.learner.error = "warn")
set.seed(126)
bmr1 = benchmark(lrns, iris.task, rdesc, measures)

library(OpenML)
task.ids = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, tag = "OpenML100", estimation.procedure = "10-fold Crossvalidation")$task.id
save(task.ids, file = "./benchmark/task_ids.RData")
tasks = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, tag = "OpenML100", estimation.procedure = "10-fold Crossvalidation")

# time estimation
time.estimate = list()
for(i in seq_along(task.ids)) {
  print(i)
  task = getOMLTask(task.ids[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  time.estimate[[i]] = estimateTimeTuneRanger(task, num.threads = 10, num.trees = 2000)
  print(time.estimate[[i]])
  save(time.estimate, file = "./benchmark/time.estimate.RData")
}
load("./benchmark/time.estimate.RData")
rdesc = makeResampleDesc("RepCV", reps = 10, folds = 5)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
# benchmark
bmr = list()
configureMlr(on.learner.error = "warn")

# Choose small and big datasets
# select datasets where RF do not take longer than ...

# take only small ones first; afterwards some bigger datasets
task.ids.bmr = task.ids[which((unlist(time.estimate)-100)<60)]
cbind(time.estimate, (unlist(time.estimate)-100)<60)

tasks[which((unlist(time.estimate)-100)<60),]

for(i in seq_along(task.ids.bmr)) { # 13 datasets
  print(i)
  set.seed(200 + i)
  task = getOMLTask(task.ids.bmr[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[i]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}
load("./benchmark/bmr.RData")
# Which datasets are not super easy (AUC < 0.99) and discriminate between the algorithms?

# medium datasets (between 160 seconds and 10 minutes)
task.ids.bmr2 = task.ids[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600)]
unlist(time.estimate)[which((unlist(time.estimate)-100)>60 & (unlist(time.estimate))<600 )]

# 13 datasets
for(i in seq_along(task.ids.bmr2)) {
  print(i)
  set.seed(200 + i)
  task = getOMLTask(task.ids.bmr2[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[length(bmr) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}
load("./benchmark/bmr.RData")

# big datasets (between 10 minutes and 1 hour)
task.ids.bmr3 = task.ids[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
# 9 datasets

rdesc = makeResampleDesc("CV", iters = 5)
# Hier evtl. doch ein paar Wiederholungen einbauen, da die Streuung sonst zu groß ist. 
# Zunächst einfach mal durchlaufen lassen (kann dannach hinzugefügt werden).
for(i in seq_along(task.ids.bmr3)) {
  print(i)
  # set.seed(245 + i) # 1. Durchlauf
  set.seed(345) # 2. Durchlauf
  task = getOMLTask(task.ids.bmr3[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[length(bmr) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}
load("./benchmark/bmr.RData")

# Very big datasets, 4 datasets
task.ids.bmr4 = task.ids[which((unlist(time.estimate))>=3600)]

# Analysis

library(mlr)
# Data cleaning

# caret problems
a = bmr[[30]]
a$results$Australian$caret$measures.test
# no results at all

# tuneRangerMMCE problems
a = bmr[[20]]
resis = a$results$`kr-vs-kp`$tuneRFMMCE$measures.test
resis = resis[!is.na(resis$mmce),]
names = names(bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr)
bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr = colMeans(resis)[2:6]
names(bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr) = names

# tuneRF problems
a = bmr[[4]]
resis = a$results[[1]]$tuneRF$measures.test
resis = resis[!is.na(resis$mmce),]
names = names(bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr)
bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr = colMeans(resis)[2:6]
names(bmr[[20]]$results$`kr-vs-kp`$tuneRFMMCE$aggr) = names

# mlr summary functions
bmr_tot = bmr[[1]]
for(i in 2:length(bmr))
  bmr_tot$results = c(bmr_tot$results, bmr[[i]]$results)
plotBMRSummary(bmr_tot, pretty.names = F, pointsize = 4, jitter = 0.2, measure = multiclass.au1p)


# Wilcoxon paired test
library(mlr)
data = array(NA, dim = c(length(bmr), 2, 5))
for(i in 1:length(bmr)) {
  print(i)
  res_aggr = data.frame(getBMRAggrPerformances(bmr[[i]]))
  data[i,,] = t(data.frame(res_aggr[,c(3,8) ]))
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
    resi[[i]][5,6] = max(resi[[i]][5,], na.rm = T)
  }
  # tuneRF gets the worst results if NA
  if(is.na(resi[[i]][1,7])) {
    resi[[i]][1,7] = max(resi[[i]][1,], na.rm = T)
    resi[[i]][2,7] = min(resi[[i]][2,], na.rm = T)
    resi[[i]][3,7] = max(resi[[i]][3,], na.rm = T)
    resi[[i]][4,7] = max(resi[[i]][4,], na.rm = T)
    resi[[i]][5,7] = max(resi[[i]][5,], na.rm = T)
  }
  res_aggr = res_aggr + resi[[i]]
  res_aggr_rank = res_aggr_rank + apply(resi[[i]], 1, rank)
}
res_aggr = res_aggr/length(bmr)
lrn.names = c(paste0("tuneRanger", c("MMCE", "AUC", "Brier", "Logloss")), "hyperopt", "caret", "tuneRF", "ranger")
colnames(res_aggr) = lrn.names
library(stringr)
rownames(res_aggr) = str_sub(rownames(res_aggr), start=1, end=-11)
t(res_aggr)
rownames(res_aggr) = c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime")
library(xtable)
xtable(t(res_aggr), digits = 4, caption = "Average performance results of the different algorithms for the small datasets", label = "avg_small")

# average rank matrix
res_aggr_rank = res_aggr_rank/length(bmr)

rownames(res_aggr_rank) = lrn.names
colnames(res_aggr_rank) = str_sub(colnames(res_aggr_rank), start=1, end=-11)
colnames(res_aggr_rank) = c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime")
xtable(res_aggr_rank, digits = 2, caption = "Average rank results of the different algorithms for the small datasets", label = "rank_small")

library(knitr)
rownames(res_aggr) = paste("--", c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime"))
kable(t(round(res_aggr,4)))
colnames(res_aggr_rank) = paste("--", c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime"))
kable(round(res_aggr_rank,2))


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
legend("topleft", legend = lrn.names, col = 1:ncol(time), lty = 1)

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
  legend("topleft", legend = lrn.names, col = 1:ncol(perfi), lty = 1)
}

# Compared to ranger model
perfis = list()
lrn.names2 = c(paste0("tR", c("MMCE", "AUC", "Brier", "Logloss")), "hyperopt", "caret", "tuneRF", "ranger")
perfi = matrix(NA, length(bmr),  ncol(resi[[1]]))
for(j in c(1:4)) {
  for(i in 1:30) {
    perfi[i,] = unlist(resi[[i]][j,]) - unlist(resi[[i]][j,8])
  }
  colnames(perfi) = lrn.names2
  perfis[[j]] = perfi
  
  #perfi = perfi[time_order,]
  #perfi = perfi[order(perfi[,1]),]
  print(plot(perfi[,1], type = "l", ylim = c(min(perfi, na.rm = T), max(perfi, na.rm = T)), ylab = rownames(res_aggr)[j], xlab = "Dataset number"))
  for(i in 2:ncol(time))
    lines(perfi[,i], col = i)
  legend("bottomright", legend = sapply(lrns, getLearnerId), col = 1:ncol(perfi), lty = 1)
}

measure.names = c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss")
op <- par(mfrow = c(2,2),
  oma = c(0,0,0,0) + 0.1,
  mar = c(2,2,1,0) + 0.1)


for(i in 1:4) {
  boxplot(perfis[[i]], main = measure.names[i])
  abline(0, 0, col = "red")
}
for(i in 1:4) {
  boxplot(perfis[[i]], outline = FALSE, main = measure.names[i])
  abline(0, 0, col = "red")
}


# regression

library(devtools)
load_all("../tuneRanger")

# Compare runtime and AUC/Brier Score with mlr
library(mlr)

source("./benchmark/RLearner_regr_caretRanger.R")
library(mlrHyperopt)
source("./benchmark/RLearner_regr_hyperoptRanger.R")
library(randomForest)
source("./benchmark/RLearner_regr_tuneRF.R")

lrns = list(
  makeLearner("regr.tuneRanger", id = "tuneRFMSE", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(mse))),
  makeLearner("regr.tuneRanger", id = "tuneRFMSE_mtry", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(mse), tune.parameters = "mtry")),
  makeLearner("regr.hyperoptRanger", id = "hyperopt"), 
  makeLearner("regr.caretRanger", id = "caret"), 
  makeLearner("regr.tuneRF", id = "tuneRF"), 
  makeLearner("regr.ranger", id = "ranger", par.vals = list(num.trees = 2000, num.threads = 10, respect.unordered.factors = "order"))
)


load("./benchmark/regression/regression_datasets_manual.RData")
reg = reg[reg$number.of.instances >= 500, ]
# 38 datasets
reg$number.of.features
task.ids.regr = reg$task.id

rdesc = makeResampleDesc("RepCV", reps = 2, folds = 5)
measures = list(mse, mae, medse, medae, rsq, kendalltau, spearmanrho, timetrain)
configureMlr(on.learner.error = "warn")

rdesc = makeResampleDesc("Holdout")

namen = numeric(30)
for(i in seq_along(task.ids.regr)[-1]) { # 38 datasets
  print(i)
  set.seed(200 + i)
  task = getOMLTask(task.ids.regr[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_regr = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_regr, file = "./benchmark/bmr_regr.RData")
}
bmr_regr
# Anhang



