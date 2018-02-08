makeRLearner.classif.tuneRF = function() {
  makeRLearnerClassif(
    cl = "classif.tuneRF",
    package = "randomForest",
    par.set = makeParamSet(
    ),
    properties = c("twoclass", "multiclass", "prob", "numerics", "factors", "ordered", "featimp", "weights"),
    name = "tuneRF of random forest",
    short.name = "tuneRF",
    note = ""
  )
}

trainLearner.classif.tuneRF = function(.learner, .task, .subset, .weights = NULL, classwt = NULL, cutoff, ...) {
  f = getTaskFormula(.task)
  data = getTaskData(.task, .subset, recode.target = "drop.levels")
  levs = levels(data[, getTaskTargetNames(.task)])
  n = length(levs)
  if (missing(cutoff))
    cutoff = rep(1 / n, n)
  if (!missing(classwt) && is.numeric(classwt) && length(classwt) == n && is.null(names(classwt)))
    names(classwt) = levs
  if (is.numeric(cutoff) && length(cutoff) == n && is.null(names(cutoff)))
    names(cutoff) = levs
  target = data[, getTaskTargetNames(.task)]
  indi = which(colnames(data) == getTaskTargetNames(.task))
  x = data[, -indi]
  randomForest::tuneRF(x = x, y = target, data = data, ntree = 2000, doBest = TRUE, ...)
}

predictLearner.classif.tuneRF = function(.learner, .model, .newdata, ...) {
  type = ifelse(.learner$predict.type == "response", "response", "prob")
  predict(.model$learner.model, newdata = .newdata, type = type, ...)
}