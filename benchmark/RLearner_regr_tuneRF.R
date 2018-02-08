makeRLearner.regr.tuneRF = function() {
  makeRLearnerRegr(
    cl = "regr.tuneRF",
    package = "randomForest",
    par.set = makeParamSet(
    ),
    properties = c("numerics", "factors", "ordered"),
    name = "tuneRF of random forest",
    short.name = "tuneRF",
    note = ""
  )
}

trainLearner.regr.tuneRF = function(.learner, .task, .subset, .weights = NULL, classwt = NULL, cutoff, ...) {
  f = getTaskFormula(.task)
  data = getTaskData(.task, .subset)
  target = data[, getTaskTargetNames(.task)]
  indi = which(colnames(data) == getTaskTargetNames(.task))
  x = data[, -indi]
  randomForest::tuneRF(x = x, y = target, data = data, ntree = 2000, doBest = TRUE, ...)
}

predictLearner.regr.tuneRF = function(.learner, .model, .newdata, ...) {
  predict(.model$learner.model, newdata = .newdata, type = "response", ...)
}