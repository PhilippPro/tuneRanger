#' @export
makeRLearner.classif.tuneRanger = function() {
  makeRLearnerClassif(
    cl = "classif.tuneRanger",
    package = "tuneRanger",
    par.set = makeParamSet(
      makeUntypedLearnerParam(id = "measure", default = multiclass.brier),
      makeIntegerLearnerParam(id = "iters", lower = 1L, default = 70L),
      makeIntegerLearnerParam(id = "iters.warmup", lower = 1L, default = 30L),
      makeNumericLearnerParam(id = "time.budget", lower = 1L),
      makeIntegerLearnerParam(id = "num.threads", lower = 1L, when = "both", tunable = FALSE),
      makeIntegerLearnerParam(id = "num.trees", lower = 1L, default = 500L),
      makeUntypedLearnerParam(id = "tune.parameters", default = c("mtry", "min.node.size", "sample.fraction")),
      makeUntypedLearnerParam(id = "parameters", default = list(replace = FALSE, respect.unordered.factors = "order"))
      ),
    properties = c("twoclass", "multiclass", "prob", "numerics", "factors", "ordered", "weights"),
    name = "Random Forests",
    short.name = "tuneRanger",
    note = "By default, internal parallelization is switched off (`num.threads = 1`), `verbose` output is disabled, `respect.unordered.factors` is set to `order`. All settings are changeable."
  )
}

#' @export
trainLearner.classif.tuneRanger = function(.learner, .task, .subset, .weights = NULL, ...) {
  tuneRanger::tuneRanger(task = subsetTask(.task, .subset), build.final.model = TRUE, ...)$model
}

#' @export
predictLearner.classif.tuneRanger = function(.learner, .model, .newdata, ...) {
  model = .model$learner.model$learner.model
  p = predict(object = model, data = .newdata, ...)
  if (.learner$predict.type == "response") {
    classes = factor(colnames(p$predictions)[apply(p$predictions, 1, which.max)], levels = colnames(p$predictions))
    return(classes)
  } else {
    return(p$predictions)
  }
}

#' #' @export
#' getOOBPredsLearner.classif.ranger = function(.learner, .model) {
#'   .model$learner.model$predictions
#' }
#' 
#' #' @export
#' getFeatureImportanceLearner.classif.ranger = function(.learner, .model, ...) {
#'   has.fiv = .learner$par.vals$importance
#'   if (is.null(has.fiv) || has.fiv == "none") {
#'     stop("You must set the learners parameter value for importance to
#'       'impurity' or 'permutation' to compute feature importance")
#'   }
#'   mod = getLearnerModel(.model)
#'   ranger::importance(mod)
#' }

  