options(java.parameters = "-Xmx16000m") # should avoid java gc overhead

library(OpenML)
# devtools::install_github("mlr-org/mlr")
library(mlr)
library(tuneRanger)
# devtools::install_github("ja-thomas/autoxgboost")
library(autoxgboost)

load("./benchmark/data/datasets.RData")
reg = rbind(reg, reg_syn)

bmr_tune <- list()
lrns = list(
  makeLearner("regr.ranger", num.trees = 500, num.threads = 5),
  makeLearner("regr.tuneRanger", num.threads = 5, time.budget = 3600),
  makeLearner("regr.autoxgboost", nthread = 5),
  makeLearner("regr.liquidSVM", threads = 5)
)
rdesc <- makeResampleDesc("CV", iters= 5)

for(i in c(1:nrow(reg))[-c(1:28)]) {
  print(i)
  task <- convertOMLTaskToMlr(getOMLTask(task.id = reg$task.id[i]))$mlr.task
  set.seed(i + 321)
  bmr_tune[[i]] <- benchmark(lrns, task, rdesc, keep.pred = FALSE, models = FALSE, measures = list(mse, mae, medse, medae, rsq, spearmanrho, kendalltau, timetrain))
  save(bmr_tune, file = "./benchmark/data/bmr_tune.RData")
}

load("./benchmark/data/bmr_tune.RData")

# kendalls tau optimieren?
# publish them on the tuneRanger github page
# h2O?, autosklearn, AutoWeka?
# tuneRanger mit weniger Iterationen ausprobieren (?) (Achtung mit Overfitting...)


# if less than 20 percent NA, impute by the mean of the other iterations
# for(i in seq_along(bmr_tune)[-c(2, 28)]) {
#   for(j in 1:nr.learners) {
#     print(paste(i,j))
#     for(k in 2:nr.measures) {
#       na.percentage = mean(is.na(bmr_tune[[i]]$results[[1]][[j]]$measures.test[k]))
#       if(na.percentage > 0 & na.percentage <= 0.2) {
#         resis = unlist(bmr_tune[[i]]$results[[1]][[j]]$measures.test[k])
#         bmr_tune[[i]]$results[[1]][[j]]$aggr[k-1] = mean(resis, na.rm = T)
#       }
#     }
#   }
# }



# Analysis
load("./benchmark/data/bmr_tune.RData")
nr.learners = length(bmr_tune[[1]]$learners)
nr.measures = length(bmr_tune[[1]]$measures)
bmr_tune[[2]] = NULL
bmr_tune[[27]] = NULL

nr.learners = length(bmr_tune[[1]]$learners)
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr_tune[[1]]))

for(i in 1:length(bmr_tune)) {
  if(!is.null(bmr_tune[[i]])) {
    resi[[i]] = data.frame(getBMRAggrPerformances(bmr_tune[[i]]))
    for(j in 1:nr.learners) {
      print(paste(i,j))
      for(k in 1:8) {
        if(is.na(resi[[i]][k,j])) {
          if(k %in% c(1:4, 8)) {
            resi[[i]][k,j] = max(resi[[i]][k,], na.rm = T)
          } else {
            resi[[i]][k,j] = min(resi[[i]][k,], na.rm = T)
          }
        }
      }
    }
    if(i == 1) {
      res_aggr = resi[[1]]
      res_aggr_rank = apply(resi[[1]], 1, rank)
    } else {
      res_aggr = res_aggr + resi[[i]]
      res_aggr_rank = res_aggr_rank + apply(resi[[i]], 1, rank)
    }
  }
}
res_aggr = res_aggr/(length(bmr_tune))
res_aggr_rank = res_aggr_rank/(length(bmr_tune))

#for(i in 3:length(bmr_tune)) {
#  resi[[i]] = round(cbind(resi[[i]], data.frame(getBMRAggrPerformances(bmr_autoxgboost[[i]]))), 4)
#  resi[[i]] = round(cbind(resi[[i]], data.frame(getBMRAggrPerformances(bmr_liquidSVM[[i]]))), 4)
#}

lrn.names = sub('.*\\.', '', colnames(res_aggr))  
meas.names = sub("\\..*", "", rownames(res_aggr))

tab1 = round(res_aggr[5:6,], 3)
tab2 = t(round(res_aggr_rank[,5:6], 3))

colnames(tab1) = colnames(tab2) = lrn.names
rownames(tab1) = rownames(tab2) = c("R-squared", "Spearman Rho")

library(xtable)
xtable(tab1)
xtable(tab2)

# R-squared

plot_results = function(j, log = FALSE, ylab = NULL, legend.pos = NULL) {
  ranger_res = matrix(NA, length(resi), 4)
  ranger_res[1, ] = as.numeric(resi[[1]][j, ])
  for(i in 1:length(resi))
    ranger_res[i, ] = as.numeric(resi[[i]][j, ])
  
  ranger_res = ranger_res[order(ranger_res[,1]),]
  if(is.null(ylab))
    ylab = sub("\\..*", "", rownames(resi[[1]])[j])
  if(is.null(legend.pos))
    legend.pos = "topleft"
  if(log) {
    plot(ranger_res[,1], type = "l", xlab = paste("Datasets ordered by", ylab, "of ranger"), ylab = ylab, log = "y", lwd = 2, lty = 2, ylim = range(ranger_res))
  } else {
    plot(ranger_res[,1], type = "l", xlab = paste("Datasets ordered by", ylab, "of ranger"), ylab = ylab, ylim = c(-0.05,1), lwd = 2, lty = 2)
  }
  lines(ranger_res[,2], col = "blue", lwd = 2)
  lines(ranger_res[,3], col = "red")
  lines(ranger_res[,4], col = "green")
  legend(legend.pos, legend = lrn.names, col = c("black", "blue", "red", "green"), lwd = c(2,2,1,1), lty = c(2,1,1,1))
}

plot_results(5, ylab = "R-Squared")
plot_results(6, ylab = "Spearman-Rho")
plot_results(7)
plot_results(1, log = TRUE)



pdf("./benchmark/figure/rsq_results.pdf", height = 4)
par(mar = c(4, 4, 1, 2) + 0.1)
plot_results(5, ylab = "R-Squared")
dev.off()

pdf("./benchmark/figure/spearman_results.pdf", height = 4)
par(mar = c(4, 4, 1, 2) + 0.1)
plot_results(6, ylab = "Spearmans-Rho", legend.pos = "bottomright")
dev.off()

pdf("./benchmark/figure/time_results.pdf", height = 4)
par(mar = c(4, 4, 1, 2) + 0.1)
plot_results(8, ylab = "Training time in seconds", legend.pos = "bottomright", log = TRUE)
dev.off()

# mit seed nochmal neu laufen lassen
# Ergebnisse mit Janek besprechen; insbesondere Laufzeitbeschränkung...