
# Finally make more parameters possible. This is just a test version.
makeRLearner.classif.tuneRF = function() {
  makeRLearnerClassif(
    cl = "classif.tuneRF",
    package = "tuneRF",
    par.set = makeParamSet(
      makeUntypedLearnerParam(id = "measure", default = multiclass.brier),
      makeIntegerLearnerParam(id = "iters", lower = 1L, default = 100L),
      makeIntegerLearnerParam(id = "num.threads", lower = 1L, when = "both", tunable = FALSE),
      makeIntegerLearnerParam(id = "num.trees", lower = 1L, default = 500L)
    ),
    properties = c("twoclass", "multiclass", "prob", "numerics", "factors", "ordered", "featimp", "weights", "oobpreds"),
    name = "Random Forests",
    short.name = "tuneRF",
    note = "By default, internal parallelization is switched off (`num.threads = 1`), `verbose` output is disabled, `respect.unordered.factors` is set to `TRUE`. All settings are changeable.",
    callees = "tuneRF"
  )
}

# task, measure = NULL, iters = 100, num.threads = NULL, num.trees = 1000, 
# parameters = list(replace = TRUE, respect.unordered.factors = TRUE

trainLearner.classif.tuneRF = function(.learner, .task, .subset, .weights = NULL, ...) {
  tuneRF::tuneRF(task = subsetTask(.task, .subset), build.final.model = TRUE, ...)$model
}

predictLearner.classif.tuneRF = function(.learner, .model, .newdata, ...) {
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

  