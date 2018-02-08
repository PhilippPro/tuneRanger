# make own parameter configuration with 2000 trees; the rest is the default of the package
par.set.mlrHyperopt = makeParamSet(
  makeIntegerParam(
    id = "mtry",
    lower = 1,
    upper = expression(p),
    default = expression(round(p^0.5))),
  makeIntegerParam(
    id = "min.node.size",
    lower = 1,
    upper = 10,
    default = 1),
  keys = c("p"))
par.config.mlrHyperopt = makeParConfig(
  par.set = par.set.mlrHyperopt,
  par.vals = list(num.trees = 2000, num.threads = 10),
  learner.name = "ranger"
)

makeRLearner.regr.hyperoptRanger = function() {
  makeRLearnerRegr(
    cl = "regr.hyperoptRanger",
    package = "mlrHyperopt",
    par.set = makeParamSet(
    ),
    properties = c("numerics", "factors", "ordered"),
    name = "Random Forests",
    short.name = "hyperoptRanger",
    note = "By default, internal parallelization is switched off (`num.threads = 1`), `verbose` output is disabled, `respect.unordered.factors` is set to `TRUE`. All settings are changeable."
  )
}

trainLearner.regr.hyperoptRanger = function(.learner, .task, .subset, .weights = NULL, ...) {
  res = hyperopt(.task, learner = "regr.ranger", par.config = par.config.mlrHyperopt)
  lrn = setPredictType(res$learner, .learner$predict.type)
  mlr::train(lrn, .task, subset = .subset, weights = .weights)
}

predictLearner.regr.hyperoptRanger = function(.learner, .model, .newdata, ...) {
  model = .model$learner.model$learner.model
  p = predict(object = model, data = .newdata, ...)
  return(p$predictions)
}