

#' Title
#'
#' @param formula 
#' @param data 
#' @param measure 
#' @param num.threads 
#' @param num.trees 
#' @param replace 
#' @param iters 
#' @param save.file.path 
#'
#' @return
#' @export
#'
#' @examples
tuneRanger = function(formula, data, measure = measureMSE, num.threads = 1, num.trees = 1000, replace = TRUE, iters = 100, save.file.path = "./optpath.RData") {
  performan = function(x) {
    pred = ranger(formula = formula, data = data,  mtry = x$mtry, min.node.size = x$min.node.size, 
      sample.fraction = x$sample.fraction, replace = TRUE, num.trees = num.trees, 
      respect.unordered.factors = TRUE, num.threads = num.threads)$predictions
    target = all.vars(formula)[1]
    return(measure(pred, train[, target]))
  }
  
  # Its ParamSet
  ps = makeParamSet(
    makeIntegerParam("min.node.size", lower = 1, upper = round(nrow(data)/4)),
    makeNumericParam("sample.fraction", lower = 0.2, upper = 0.9),
    makeIntegerParam("mtry", lower = 1, upper = ncol(data))
  )
  
  # Budget
  f.evals = iters
  mbo.init.design.size = 30
  
  # Focus search
  infill.opt = "focussearch"
  mbo.focussearch.points = iters
  mbo.focussearch.maxit = 3
  mbo.focussearch.restarts = 3
  
  library(mlrMBO)
  # The final SMOOF objective function
  objFun = makeMultiObjectiveFunction(
    name = "reg",
    fn = performan,
    par.set = ps,
    has.simple.signature = FALSE,
    noisy = TRUE,
    n.objectives = 1,
    minimize = TRUE
  )
  
  # Build the control object
  method = "parego"
  if (method == "parego") {
    mbo.prop.points = 1
    mbo.crit = "cb"
    parego.crit.cb.pi = 0.5
  }
  
  control = makeMBOControl(n.objectives = 1L, propose.points = mbo.prop.points, impute.y.fun = function(x, y, opt.path) 0.7, 
    save.on.disk.at = 1:10, save.file.path = save.file.path)
  control = setMBOControlTermination(control, max.evals =  f.evals, iters = 300)
  control = setMBOControlInfill(control, #opt = infill.opt,
    opt.focussearch.maxit = mbo.focussearch.maxit,
    opt.focussearch.points = mbo.focussearch.points,
    opt.restarts = mbo.focussearch.restarts)
  
  mbo.learner = makeLearner("regr.randomForest", predict.type = "se")
  
  design = generateDesign(mbo.init.design.size, getParamSet(objFun), fun = lhs::maximinLHS)
  
  set.seed(123)
  result = mbo(fun = objFun, design = design, learner = mbo.learner, control = control)
  
  res = data.frame(result$opt.path)
  res
}





measureMSE = function (truth, response) {
  mean((response - truth)^2)
}

