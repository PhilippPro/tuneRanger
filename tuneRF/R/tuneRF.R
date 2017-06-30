library(ranger)
library(mlrMBO)
library(mlr)
library(readr)

load("C:/Promotion/Kaggle/Sberbank/data/preproc_ranger.RData")

# x = list(mtry = 1, min.node.size = 1, sample.fraction = 1, replace = TRUE)
#train$TARGET = as.factor(train$TARGET)

performan = function(x) {
  pred <- ranger(formula = price_doc ~ . , data = train,  mtry = x$mtry, min.node.size = x$min.node.size, 
    sample.fraction = x$sample.fraction, replace = x$replace, num.trees = 1000, 
    respect.unordered.factors = TRUE, num.threads = 6)$predictions
  return(mlr::measureRMSLE(pred, train$price_doc))
}

x = list(replace = TRUE, min.node.size = 3, sample.fraction = 0.5, mtry = 100)
# Its ParamSet
ps = makeParamSet(
  makeLogicalParam("replace"),
  makeIntegerParam("min.node.size", lower = 1, upper = 3000),
  makeNumericParam("sample.fraction", lower = 0.2, upper = 0.9),
  makeIntegerParam("mtry", lower = 1, upper = 200)
)

# Budget
f.evals = 100
mbo.init.design.size = 30

# Focus search
infill.opt = "focussearch"
mbo.focussearch.points = 100
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
  save.on.disk.at = 1:10, save.file.path = "C:/Promotion/Kaggle/Sberbank/results/optpath.RData")
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
res[res$y < 0.48,]

save(res, file = "results_mbo_ranger.RData")
res[res$y < 0.462,]
# replace min.node.size sample.fraction mtry         y dob eol error.message exec.time        cb error.model
# 28    TRUE             8       0.8668154  104 0.4619139   0  NA          <NA>    417.39        NA        <NA>
#   54    TRUE             4       0.7827807  111 0.4618559  24  NA          <NA>    435.84 0.4589545        <NA>
#   58    TRUE             2       0.7403381  130 0.4619768  28  NA          <NA>    514.59 0.4597870        <NA>
#   84    TRUE             3       0.6784268  156 0.4619376  54  NA          <NA>    554.16 0.4605459        <NA>
#   train.time  prop.type propose.time          se      mean lambda
# 28         NA initdesign           NA          NA        NA     NA
# 54       0.01  infill_cb         0.16 0.006082860 0.4711202      2
# 58       0.02  infill_cb         0.17 0.007308362 0.4744038      2
# 84       0.00  infill_cb         0.20 0.006062907 0.4726718      2




set.seed(150)

ranger_params = list(
  num.threads = 8, 
  num.trees = 2000,
  sample.fraction = 0.75,
  min.node.size = 4,
  mtry = 120,
  replace = TRUE,
  respect.unordered.factors = TRUE
)
lrn = makeLearner("regr.ranger", par.vals = ranger_params)
task = makeRegrTask(id="russia", data = train, target = "price_doc", fixup.data = "warn")
mod = train(lrn, task)
pred = predict(mod, newdata = test)

submission = read.csv("C:/Promotion/Kaggle/Sberbank/data/sample_submission.csv")
head(submission)
all(test_ids == submission$id)
submission$price_doc = pred$data$response#expm1(pred$data$response)
write_csv(submission, "C:/Promotion/Kaggle/Sberbank/results/mlr_ranger_mbo_tuned.csv")