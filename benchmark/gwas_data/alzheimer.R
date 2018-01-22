
library(GenABEL)
library(ranger)

setwd("./benchmark/gwas_data")
alzheimer <- readRDS("alzheimer_gwaa.Rds")

# Run on GenABEL data: Quite fast
system.time(mod <- ranger(y ~ ., alzheimer, num.threads = 1))
ranger(y ~ ., alzheimer, mtry = 10000)

# Convert to data.frame and run: Very slow!
alzheimer_df <- data.frame(as.data.frame(alzheimer@phdata),
                           as.double(alzheimer@gtdata))
system.time(mod <- ranger(y ~ ., alzheimer_df, num.trees = 10, num.threads = 1))
# viel zu lang