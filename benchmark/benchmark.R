library(devtools)
library(OpenML)
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

rdesc = makeResampleDesc("Holdout")
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)
configureMlr(on.learner.error = "warn")
set.seed(126)
bmr1 = benchmark(lrns, iris.task, rdesc, measures)

library(OpenML)
#task.ids = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, tag = "OpenML100", estimation.procedure = "10-fold Crossvalidation")$task.id
#save(task.ids, file = "./benchmark/task_ids.RData")

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
unlist(time.estimate)[which((unlist(time.estimate)-100)<60)]
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
  set.seed(300 + i)
  task = getOMLTask(task.ids.bmr2[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr[[length(bmr) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr, file = "./benchmark/bmr.RData")
}
load("./benchmark/bmr.RData")

# big datasets (between 10 minutes and 1 hour)
task.ids.bmr3 = task.ids[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
unlist(time.estimate)[which((unlist(time.estimate))>600 & (unlist(time.estimate))<3600)]
# 9 datasets

rdesc = makeResampleDesc("CV", iters = 5)
bmr_big = list()
# Hier evtl. doch ein paar Wiederholungen einbauen, da die Streuung sonst zu groß ist. 
# Zunächst einfach mal durchlaufen lassen (kann dannach hinzugefügt werden).
for(i in seq_along(task.ids.bmr3)) {
  print(i)
  set.seed(400 + i) 
  task = getOMLTask(task.ids.bmr3[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  bmr_big[[length(bmr_big) + 1]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
  save(bmr_big, file = "./benchmark/bmr_big.RData")
}
load("./benchmark/bmr_big.RData")

# Very big datasets, 4 datasets
tasks4 = tasks[which((unlist(time.estimate))>=3600),]
task.ids.bmr4 = task.ids[which((unlist(time.estimate))>=3600)]
unlist(time.estimate)[which((unlist(time.estimate))>=3600)]

######################################################### Analysis ######################################################

load("./benchmark/bmr.RData")

# Analyse "strange" logloss results for ranger
rdesc = makeResampleDesc("Holdout")
set.seed(200)
task = getOMLTask(task.ids.bmr[13])
task = convertOMLTaskToMlr(task)$mlr.task
bmr_tuneRF = benchmark(lrns[11:12], task, rdesc, measures, keep.pred = TRUE, models = FALSE)
hist(sort(bmr_tuneRF$results$`blood-transfusion-service-center`$tuneRF$pred$data$prob.1))
hist(sort(bmr_tuneRF$results$`blood-transfusion-service-center`$ranger$pred$data$prob.1), col = "red")

library(mlr)
# Data cleaning
names = names(bmr[[1]]$results[[1]]$tuneRFMMCE$aggr)
nr.learners = length(bmr[[1]]$learners)

# if less than 20 percent NA, impute by the mean of the other iterations
for(i in seq_along(bmr)) {
  for(j in 1:nr.learners) {
    print(paste(i,j))
    na.percentage = mean(is.na(bmr[[i]]$results[[1]][[j]]$measures.test$mmce))
    if(na.percentage > 0 & na.percentage <= 0.2) {
      resis = bmr[[i]]$results[[1]][[j]]$measures.test
      bmr[[i]]$results[[1]][[j]]$aggr = colMeans(resis[!is.na(resis$mmce),])[2:6]
      names(bmr[[i]]$results[[1]][[j]]$aggr) = names
    }
  }
}

# Analysis of time
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr[[1]]))

for(i in 2:length(bmr)) {
  resi[[i]] = data.frame(getBMRAggrPerformances(bmr[[i]]))
  # caret gets no result, if NA
}

lty.vec = c(rep(1,4), c(2,3,4,5))
library(RColorBrewer)
col.vec = brewer.pal(8, "Dark2")
time = matrix(NA, length(bmr),  ncol(resi[[1]])-4)
for(i in seq_along(bmr)) {
  time[i,] = unlist(resi[[i]][5,-c(5:8)])
}
time_order = order(time[,1])
time = time[time_order,]
plot(time[,1], type = "l", ylim = c(0, max(time, na.rm = T)), ylab = "Time in seconds", xlab = "Dataset number", col = col.vec[1])
for(i in 2:ncol(time)){
  points(1:length(bmr), time[,i], col = col.vec[i], cex = 0.4)
  lines(time[,i], col = col.vec[i], lty = lty.vec[i])
}
leg.names = c("tuneRangerMMCE", "tuneRangerAUC", "tuneRangerBrier", "tuneRangerLogloss", "mlrHyperopt", "caret", "tuneRF", "ranger default")
legend("topleft", legend = leg.names, col = col.vec, lty = lty.vec)


# Descriptive Analysis
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr[[1]]))
res_aggr = resi[[1]]
res_aggr_rank = apply(resi[[1]][-c(5:8)], 1, rank)

for(i in 2:length(bmr)) {
  resi[[i]] = data.frame(getBMRAggrPerformances(bmr[[i]]))
  # models gets the worst result, if NA
  for(j in 1:12) {
    print(paste(i,j))
    if(is.na(resi[[i]][1,j])) {
      resi[[i]][1,j] = max(resi[[i]][1,], na.rm = T)
      resi[[i]][2,j] = min(resi[[i]][2,], na.rm = T)
      resi[[i]][3,j] = max(resi[[i]][3,], na.rm = T)
      resi[[i]][4,j] = max(resi[[i]][4,], na.rm = T)
      resi[[i]][5,j] = max(resi[[i]][5,], na.rm = T)
    }
  }
  res_aggr = res_aggr + resi[[i]]
  res_aggr_rank = res_aggr_rank + apply(resi[[i]][-c(5:8)], 1, rank)
}
res_aggr = res_aggr/length(bmr)

# Graphical analysis of performance
# Compared to ranger model
perfis = list()
lrn.names2 = c(paste0(c("MMCE", "AUC", "Brier", "Logloss")), "mlrHyp.", "caret", "tuneRF", "ranger")
perfi = matrix(NA, length(bmr),  ncol(resi[[1]])-4)
for(j in c(1:4)) {
  for(i in 1:length(bmr)) {
    perfi[i,] = unlist(resi[[i]][j,])[-c(5:8)] - unlist(resi[[i]][j,12])[-c(5:8)]
  }
  colnames(perfi) = lrn.names2
  perfis[[j]] = perfi
}

measure.names = c("Error rate", "AUC", "Brier score", "Logarithmic Loss")
op <- par(mfrow = c(4,2),
  oma = c(0,0,0,0) + 0.1,
  mar = c(2.5,2,1,0) + 0.1)
outline = c(TRUE, FALSE)
outlier_name = c("", "(without outliers)")
for(i in 1:4) {
  for(j in 1:2) {
    boxplot(perfis[[i]], main = paste(measure.names[i], outlier_name[j]), horizontal = F, xaxt = "n", outline = outline[j])
    axis(1, at = c(1,2,3,4,5,6,7,8), labels = FALSE, cex = 0.1, tck = -0.02)
    #axis(1, at = c(6,8), labels = FALSE, cex = 0.1, tck = -0.07)
    mtext(lrn.names2[c(1,2,3,4,5,6,7,8)], 1, line = 0.1, at = c(1,2,3,4,5.1,6,7,8), cex = 0.6)
    #mtext(lrn.names2[c(6,8)], 1, line = 0.7, at = c(6,8), cex = 0.6)
    mtext(expression(bold("tuneRanger")), 1, line = 1.2, at = 2.5, cex = 0.6)
    abline(0, 0, col = "red")
    axis(1,at=c(0.5,1,2,3,3.5,4,4.5),col="black",line=1.15,tick=T,labels=rep("",7),lwd=2,lwd.ticks=0)
  }
}

lrn.names1 = c(paste0("tuneRanger", apply(expand.grid(c("MMCE", "AUC", "Brier", "Logloss"), c("","_mtry")), 1, paste0, collapse="")), "mlrHyperopt", "caret", "tuneRF", "ranger")
lrn.names2 = c(paste0("tuneRanger", c("MMCE", "AUC", "Brier", "Logloss")), "mlrHyperopt", "caret", "tuneRF", "ranger")
colnames(res_aggr) = lrn.names1
library(stringr)
rownames(res_aggr) = str_sub(rownames(res_aggr), start=1, end=-11)
t(res_aggr)
rownames(res_aggr) = c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime")
library(xtable)
xtable(t(res_aggr), digits = 4, caption = "Average performance results of the different algorithms for the small datasets", label = "avg_small")

# average rank matrix
res_aggr_rank = res_aggr_rank/length(bmr)

rownames(res_aggr_rank) = lrn.names2
colnames(res_aggr_rank) = str_sub(colnames(res_aggr_rank), start=1, end=-11)
colnames(res_aggr_rank) = c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime")
xtable(res_aggr_rank, digits = 2, caption = "Average rank results of the different algorithms for the small datasets", label = "rank_small")

library(knitr)
rownames(res_aggr) = paste("--", c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime"))
kable(t(round(res_aggr,4)))
colnames(res_aggr_rank) = paste("--", c("Error rate", "(Multiclass) AUC", "Brier Score", "Logarithmic Loss", "Training Runtime"))
kable(round(res_aggr_rank,2))

# Tuning only mtry is clearly worse

perfis = list()
lrn.names2 = c(paste0(c("MMCE", "AUC", "Brier", "Logloss")))
perfi = matrix(NA, length(bmr), 4)
for(j in c(1:4)) {
  for(i in 1:length(bmr)) {
    perfi[i,] = unlist(resi[[i]][j,])[5:8] - unlist(resi[[i]][j,])[1:4]
  }
  colnames(perfi) = lrn.names2
  perfis[[j]] = perfi
}

lapply(lapply(perfis, colMeans), mean)

measure.names = c("Error rate", "AUC", "Brier score", "Logarithmic Loss")
op <- par(mfrow = c(4,2),
  oma = c(0,0,0,0) + 0.1,
  mar = c(2.5,2,1,0) + 0.1)
outline = c(TRUE, FALSE)
outlier_name = c("", "(without outliers)")
for(i in 1:4) {
  for(j in 1:2) {
    boxplot(perfis[[i]], main = paste(measure.names[i], outlier_name[j]), horizontal = F, xaxt = "n", outline = outline[j])
    axis(1, at = c(1,2,3,4,5,6,7,8), labels = FALSE, cex = 0.1, tck = -0.02)
    #axis(1, at = c(6,8), labels = FALSE, cex = 0.1, tck = -0.07)
    mtext(lrn.names2[c(1,2,3,4,5,6,7,8)], 1, line = 0.1, at = c(1,2,3,4,5.1,6,7,8), cex = 0.6)
    #mtext(lrn.names2[c(6,8)], 1, line = 0.7, at = c(6,8), cex = 0.6)
    mtext(expression(bold("tuneRanger")), 1, line = 1.2, at = 2.5, cex = 0.6)
    abline(0, 0, col = "red")
    axis(1,at=c(0.5,1,2,3,3.5,4,4.5),col="black",line=1.15,tick=T,labels=rep("",7),lwd=2,lwd.ticks=0)
  }
}



# Annex

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



