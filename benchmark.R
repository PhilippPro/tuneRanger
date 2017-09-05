library(caret)
train(iris[, 1:4], iris[, 5], method = "ranger")

library(mlrHyperopt)
hyperopt(iris.task, learner = "classif.ranger")

library(tuneRF)
tuneRF(iris.task)

# Compare runtime and AUC/Brier Score with mlr

library(mlr)

lrns = list(makeLearner("classif.tuneRF", predict.type = "prob", 
  par.vals = list(num.trees = 2000, measure = list(multiclass.brier))), 
  makeLearner("classif.ranger", par.vals = list(num.trees = 2000, respect.unordered.factors = TRUE))
)
lrn = makeLearner("classif.tuneRF", predict.type = "prob")
lrn = makeLearner("classif.ranger", predict.type = "prob", par.vals = list(num.trees = 200))
# At the moment only probability prediction possible!!
mod = train(lrn, iris.task)
pred = predict(mod, task = iris.task)

library(OpenML)
tasks = listOMLTasks(number.of.classes = 2L, number.of.missing.values = 0, 
  data.tag = "study_14", estimation.procedure = "10-fold Crossvalidation")
task.ids = tasks$task.id

rdesc = makeResampleDesc("RepCV", reps = 2, folds = 2)
measures = list(mmce, multiclass.au1p, multiclass.brier, logloss, timetrain)

bmr = list()
for(i in seq_along(task.ids)) {
  task = getOMLTask(task.ids[i])
  task = convertOMLTaskToMlr(task)
  bmr[[i]] = benchmark(lrns, task, rdesc, measures, keep.pred = FALSE, models = FALSE)
}

