#' @export
makeRLearner.surv.tuneMtryFast = function() {
  makeRLearnerSurv(
    cl = "surv.tuneMtryFast",
    package = "tuneRanger",
    par.set = makeParamSet(
      makeIntegerLearnerParam(id = "mtryStart", lower = 1L),
      makeIntegerLearnerParam(id = "num.treesTry", lower = 1L, default = 50L),
      makeNumericLearnerParam(id = "stepFactor", lower = 0L),
      makeNumericLearnerParam(id = "improve", lower = 0L),
      makeLogicalLearnerParam(id = "trace", default = TRUE),
      makeIntegerLearnerParam(id = "min.node.size", lower = 1L),
      makeLogicalLearnerParam(id = "replace", default = TRUE),
      makeNumericLearnerParam(id = "sample.fraction", lower = 0L, upper = 1L),
      makeNumericVectorLearnerParam(id = "split.select.weights", lower = 0, upper = 1),
      makeUntypedLearnerParam(id = "always.split.variables"),
      makeDiscreteLearnerParam("respect.unordered.factors", values = c("ignore", "order", "partition"), default = "ignore"),
      makeDiscreteLearnerParam(id = "importance", values = c("none", "impurity", "permutation"), default = "none", tunable = FALSE),
      makeLogicalLearnerParam(id = "write.forest", default = TRUE, tunable = FALSE),
      makeLogicalLearnerParam(id = "scale.permutation.importance", default = FALSE, requires = quote(importance == "permutation"), tunable = FALSE),
      makeIntegerLearnerParam(id = "num.threads", lower = 1L, when = "both", tunable = FALSE),
      makeLogicalLearnerParam(id = "save.memory", default = FALSE, tunable = FALSE),
      makeLogicalLearnerParam(id = "verbose", default = TRUE, when = "both", tunable = FALSE),
      makeIntegerLearnerParam(id = "seed", when = "both", tunable = FALSE),
      makeDiscreteLearnerParam(id = "splitrule", values = c("gini", "extratrees"), default = "gini"),
      makeIntegerLearnerParam(id = "num.random.splits", lower = 1L, default = 1L, requires = quote(splitrule == "extratrees")),
      makeLogicalLearnerParam(id = "keep.inbag", default = FALSE, tunable = FALSE)
    ),
    properties = c("numerics", "factors", "ordered",  "weights", "prob"),
    name = "tuneMtryFast for ranger",
    short.name = "tuneMtryFast",
    note = ""
  )
}

#' @export
trainLearner.surv.tuneMtryFast = function(.learner, .task, .subset, .weights = NULL, classwt = NULL, cutoff, ...) {
  f = getTaskFormula(.task)
  target = getTaskTargetNames(.task)
  f = sprintf("Surv(%s, %s) ~ .", target[1L], target[2L])
  data = getTaskData(.task, .subset)
  tuneRanger::tuneMtryFast(formula = f, data = data, num.treesTry = 50, doBest = TRUE, case.weights = .weights, ...)
}

#' @export
predictLearner.surv.tuneMtryFast = function(.learner, .model, .newdata, ...) {
  if (predict.type == "response") {
    p = predict(object = .model$learner.model, data = .newdata)
    rowMeans(p$chf)
  } else {
    p = predict(object = .model$learner.model, data = .newdata)
    preds = rowMeans(p$chf)
    train.times = sort(unique(c(0, .model$learner.model$times)))
    ptemp = p$survival
    pos = prodlim::sindex(jump.times = p$unique.death.times, eval.times = train.times)
    probs = cbind(1, ptemp)[, pos + 1, drop = FALSE]
    list(preds = preds, probs = probs, train.times = train.times)
  }
}
