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
  data = getTaskData(.task, subset = .subset, target.extra = TRUE)
  levels(data$target) = paste0("X", levels(data$target))
  caret::train(data$data, data$target, method = "ranger", weights = .weights, num.trees = 2000, num.threads = 10, trControl = caret::trainControl(classProbs = (.learner$predict.type == "prob")))
}

predictLearner.classif.caretRanger = function(.learner, .model, .newdata, ...) {
  model = .model$learner.model
  type = ifelse(.learner$predict.type == "prob", "prob", "raw")
  p = predict(object = model, newdata = .newdata, type = type, ...)
  if (type == "prob") {
    colnames(p) = substr(colnames(p), 2, 1000)
    p = as.matrix(p)
  } else {
    levels(p) = substr(levels(p), 2, 1000) 
  }
  return(p)
}