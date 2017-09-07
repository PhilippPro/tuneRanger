makeRLearner.classif.caretRanger = function() {
  makeRLearnerClassif(
    cl = "classif.caretRanger",
    package = "caret",
    par.set = makeParamSet(
    ),
    properties = c("twoclass", "multiclass", "prob", "numerics", "factors", "ordered", "featimp", "weights"),
    name = "Random Forests",
    short.name = "caretRanger",
    note = "By default, internal parallelization is switched off (`num.threads = 1`), `verbose` output is disabled, `respect.unordered.factors` is set to `TRUE`. All settings are changeable.",
    callees = "caretRanger"
  )
}

trainLearner.classif.caretRanger = function(.learner, .task, .subset, .weights = NULL, ...) {
  target = getTaskTargets(iris.task)
  data = getTaskData(.task, subset = .subset, target.extra = TRUE)
  caret::train(data$data, data$target, method = "ranger", weights = .weights, num.trees = 2000, trControl = trainControl(classProbs = (.learner$predict.type == "prob")))
}

predictLearner.classif.caretRanger = function(.learner, .model, .newdata, ...) {
  model = .model$learner.model$finalModel
  p = predict(object = model, data = .newdata, ...)
  return(p$predictions)
}