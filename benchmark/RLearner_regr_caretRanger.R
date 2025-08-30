makeRLearner.regr.caretRanger = function() {
  makeRLearnerRegr(
    cl = "regr.caretRanger",
    package = "caret",
    par.set = makeParamSet(
    ),
    properties = c("numerics", "factors", "ordered"),
    name = "Random Forests",
    short.name = "caretRanger",
    note = "By default, internal parallelization is switched off (`num.threads = 1`), `verbose` output is disabled, `respect.unordered.factors` is set to `TRUE`. All settings are changeable."
  )
}

trainLearner.regr.caretRanger = function(.learner, .task, .subset, .weights = NULL, ...) {
  data = getTaskData(.task, subset = .subset, target.extra = TRUE)
  caret::train(data$data, data$target, method = "ranger", weights = .weights, num.trees = 2000, num.threads = 10)
}

predictLearner.regr.caretRanger = function(.learner, .model, .newdata, ...) {
  model = .model$learner.model
  p = predict(object = model, newdata = .newdata, ...)
  return(p)
}