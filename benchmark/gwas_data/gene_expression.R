library(ranger)

#source("https://bioconductor.org/biocLite.R")
#biocLite(c("cancerdata"))

# Breast cancer data
library(cancerdata) # Bioconductor!
data("VEER1")
breastcancer <- as.data.frame(VEER1)
breastcancer$info <- NULL
breastcancer <- breastcancer[rowSums(is.na(breastcancer)) == 0, ]

system.time(mod <- ranger(class ~ ., breastcancer, num.threads = 1))
# 1 second with default mtry; 3.5 seconds with mtry = 1000

# Leukemia data 
library(SIS) # CRAN
data("leukemia.train")
data("leukemia.test")
leukemia <- rbind(leukemia.train, leukemia.test)
leukemia$V7130 <- factor(leukemia$V7130)

system.time(mod <- ranger(V7130 ~ ., leukemia, num.threads = 1))
# 1.5 second with default mtry; 2.6 seconds with mtry = 1000

# Benchmark these datasets
library(mlr)
library(tuneRanger)
library(mlrHyperopt)
source("./benchmark/RLearner_classif_hyperoptRanger.R")
source("./benchmark/RLearner_classif_caretRanger.R")
library(randomForest)
source("./benchmark/RLearner_classif_tuneRF.R")

breast.task = makeClassifTask(id = "breast", data = breastcancer, target = "class")
leukemia.task = makeClassifTask(id = "leukemia", data = leukemia, target = "V7130")

lrns = list(
  makeLearner("classif.tuneRanger", id = "tuneRFMMCE", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(mmce))),
  makeLearner("classif.tuneRanger", id = "tuneRFAUC", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(multiclass.au1p))),
  makeLearner("classif.tuneRanger", id = "tuneRFBrier", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(multiclass.brier))), 
  makeLearner("classif.tuneRanger", id = "tuneRFLogloss", predict.type = "prob", 
    par.vals = list(num.trees = 2000, num.threads = 10, measure = list(logloss))), 
  makeLearner("classif.hyperoptRanger", id = "hyperopt", predict.type = "prob"), 
  makeLearner("classif.caretRanger", id = "caret", predict.type = "prob"), 
  makeLearner("classif.tuneRF", id = "tuneRF", predict.type = "prob"), 
  makeLearner("classif.ranger", id = "ranger", par.vals = list(num.trees = 2000, num.threads = 10), predict.type = "prob")
)

rdesc = makeResampleDesc("RepCV", reps = 10, folds = 5)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)

bmr_gene = list()
bmr_gene[[1]] = benchmark(lrns, breast.task, rdesc, measures)
bmr_gene[[2]] = benchmark(lrns, leukemia.task, rdesc, measures)
save(bmr_gene, file = "./benchmark/bmr_gene.RData")
