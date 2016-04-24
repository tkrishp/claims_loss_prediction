setwd("~/github/claims_loss_prediction")

train = read.csv("train.csv")
test = read.csv("test.csv")

randomSeed = 1337
set.seed(randomSeed)

library(gbm)

LogLossBinary = function(actual, predicted, eps = 1e-15) {  
  predicted = pmin(pmax(predicted, eps), 1-eps)  
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

train$RowID = NULL
train$Model = NULL
train$age_at_ins = train$CalendarYear - train$ModelYear + 1
train$CalendarYear = NULL
train$ModelYear = NULL

test$Model = NULL
test$age_at_ins = test$CalendarYear - test$ModelYear + 1
test$CalendarYear = NULL
test$ModelYear = NULL

dataSubsetProportion = .2
randomRows = sample(1:nrow(train), floor(nrow(train) * dataSubsetProportion))
trainingHoldoutSet = train[randomRows, ]
trainingNonHoldoutSet = train[!(1:nrow(train) %in% randomRows), ]

gbmWithCrossValidation = gbm(formula = Response ~ ., distribution = "bernoulli", data = trainingNonHoldoutSet,
                             n.trees = 2000, shrinkage = .1, n.minobsinnode = 200,cv.folds = 5, n.cores = 1)
bestTreeForPrediction = gbm.perf(gbmWithCrossValidation)
gbmHoldoutPredictions = predict(object = gbmWithCrossValidation, newdata = trainingHoldoutSet, 
                                n.trees = bestTreeForPrediction, type = "response")
print(paste(LogLossBinary(train$Response[randomRows], gbmHoldoutPredictions), "Holdout Log Loss"))
# 0.575160633150851
gbmTestPredictions = predict(object = gbmWithCrossValidation, newdata = test, n.trees = 1000, type = "response")
outputDataSet = data.frame("RowID" = test$RowID, "ProbabilityOfResponse" = gbmTestPredictions)
write.csv(outputDataSet, "gbm_2.csv", row.names = FALSE)
# public leaderboard: 0.58019 (position # 10)

# Remove correlated variables: Var1, Var2, Var3
trainingHoldoutSet$Var1 = NULL
trainingHoldoutSet$Var2 = NULL
trainingHoldoutSet$Var3 = NULL
trainingNonHoldoutSet$Var1 = NULL
trainingNonHoldoutSet$Var2 = NULL
trainingNonHoldoutSet$Var3 = NULL

gbmWithCrossValidation = gbm(formula = Response ~ ., distribution = "bernoulli", data = trainingNonHoldoutSet,
                             n.trees = 2000, shrinkage = .1, n.minobsinnode = 200, cv.folds = 5, n.cores = 1)
bestTreeForPrediction = gbm.perf(gbmWithCrossValidation)
gbmHoldoutPredictions = predict(object = gbmWithCrossValidation, newdata = trainingHoldoutSet, 
                                n.trees = bestTreeForPrediction, type = "response")
print(paste(LogLossBinary(train$Response[randomRows], gbmHoldoutPredictions), "Holdout Log Loss"))
# 0.57474346945061
gbmTestPredictions = predict(object = gbmWithCrossValidation, newdata = test, n.trees = 1000, type = "response")
outputDataSet = data.frame("RowID" = test$RowID, "ProbabilityOfResponse" = gbmTestPredictions)
write.csv(outputDataSet, "gbm_3.csv", row.names = FALSE)
# public leaderboard: 0.58034 (did not improve previous score of 0.58019)


####  further modifications to address correlation #### 
# Remove Var6 (it is correlated with Var4, Var5)
# Remove Var8 (it is correlated with Var4)
trainingHoldoutSet$Var6 = NULL
trainingHoldoutSet$Var8 = NULL
trainingNonHoldoutSet$Var6 = NULL
trainingNonHoldoutSet$Var8 = NULL

gbmWithCrossValidation = gbm(formula = Response ~ ., distribution = "bernoulli", data = trainingNonHoldoutSet,
                             n.trees = 2000, shrinkage = .1, n.minobsinnode = 200, cv.folds = 5, n.cores = 1)
bestTreeForPrediction = gbm.perf(gbmWithCrossValidation)
gbmHoldoutPredictions = predict(object = gbmWithCrossValidation, newdata = trainingHoldoutSet, 
                                n.trees = bestTreeForPrediction, type = "response")
print(paste(LogLossBinary(train$Response[randomRows], gbmHoldoutPredictions), "Holdout Log Loss"))
# 0.574628049336264
gbmTestPredictions = predict(object = gbmWithCrossValidation, newdata = test, n.trees = 1500, type = "response")
outputDataSet = data.frame("RowID" = test$RowID, "ProbabilityOfResponse" = gbmTestPredictions)
write.csv(outputDataSet, "gbm_4.csv", row.names = FALSE)
# public leaderboard: 0.58008 (improved from previous score of 0.58019; no change in position # 10)

#### add interaction depth to handle 2 way interactions ####
gbmWithCrossValidation = gbm(formula = Response ~ ., distribution = "bernoulli", data = trainingNonHoldoutSet,
                             n.trees = 2000, shrinkage = .1, n.minobsinnode = 200, cv.folds = 5, 
                             interaction.depth = 2, n.cores = 4)
bestTreeForPrediction = gbm.perf(gbmWithCrossValidation)
gbmHoldoutPredictions = predict(object = gbmWithCrossValidation, newdata = trainingHoldoutSet, 
                                n.trees = bestTreeForPrediction, type = "response")
print(paste(LogLossBinary(train$Response[randomRows], gbmHoldoutPredictions), "Holdout Log Loss"))
# 0.574413289425078
gbmTestPredictions = predict(object = gbmWithCrossValidation, newdata = test, n.trees = 2000, type = "response")
outputDataSet = data.frame("RowID" = test$RowID, "ProbabilityOfResponse" = gbmTestPredictions)
write.csv(outputDataSet, "gbm_5.csv", row.names = FALSE)
# public leaderboard: 0.58272 (did not improve from previous score of 0.58019)


#### model with probabilities for Make variable
# compute prior probabilities for each make
x = left_join((train %>% group_by(Make) %>% summarize(count = n())),
              (train %>% filter(Response == 1) %>% group_by(Make) %>% summarize(ins_filed = n())),
              by = 'Make')
x[is.na(x)] = 0.391285
x$prior_prob = x$ins_filed/x$count
train$make_prob = inner_join(train, x, by = 'Make') %>% select(prior_prob)
train = inner_join(train, x, by = 'Make') %>% select(-count, -ins_filed)

test = left_join(test, x, by = 'Make') %>% select(-count, -ins_filed)
test[is.na(test$prior_prob)] = 0.391285

trainingHoldoutSet = train[randomRows, ]
trainingNonHoldoutSet = train[!(1:nrow(train) %in% randomRows), ]

gbmWithCrossValidation = gbm(formula = Response ~ ., distribution = "bernoulli", data = trainingNonHoldoutSet,
                             n.trees = 2000, shrinkage = .1, n.minobsinnode = 200, cv.folds = 5, 
                             interaction.depth = 2, n.cores = 4)
bestTreeForPrediction = gbm.perf(gbmWithCrossValidation)
gbmHoldoutPredictions = predict(object = gbmWithCrossValidation, newdata = trainingHoldoutSet, 
                                n.trees = bestTreeForPrediction, type = "response")
print(paste(LogLossBinary(train$Response[randomRows], gbmHoldoutPredictions), "Holdout Log Loss"))
# 0.580375716785375
gbmTestPredictions = predict(object = gbmWithCrossValidation, newdata = test, n.trees = 2000, type = "response")
outputDataSet = data.frame("RowID" = test$RowID, "ProbabilityOfResponse" = gbmTestPredictions)
write.csv(outputDataSet, "gbm_7.csv", row.names = FALSE)
# public leaderboard: 0.58242 (did not improve from previous score of 0.58019)


#### use only K, AU, Y make ####
randomSeed = 1337
set.seed(randomSeed)

train = read.csv("train.csv")
test = read.csv("test.csv")

train$RowID = NULL
train$Model = NULL
train$age_at_ins = train$CalendarYear - train$ModelYear + 1
train$CalendarYear = NULL
train$ModelYear = NULL

test$Model = NULL
test$age_at_ins = test$CalendarYear - test$ModelYear + 1
test$CalendarYear = NULL
test$ModelYear = NULL

train$Make = ifelse(train$Make == 'K', as.character(train$Make), 
             ifelse(train$Make == 'AU', as.character(train$Make),
             ifelse(train$Make == 'Y', as.character(train$Make), 'OTH')))
train$Make = as.factor(train$Make)

test$Make = ifelse(test$Make == 'K', as.character(test$Make), 
            ifelse(test$Make == 'AU', as.character(test$Make),
            ifelse(test$Make == 'Y', as.character(test$Make), 'OTH')))
test$Make = as.factor(test$Make)

dataSubsetProportion = .2
randomRows = sample(1:nrow(train), floor(nrow(train) * dataSubsetProportion))
trainingHoldoutSet = train[randomRows, ]
trainingNonHoldoutSet = train[!(1:nrow(train) %in% randomRows), ]

trainingHoldoutSet$Var1 = NULL
trainingHoldoutSet$Var2 = NULL
trainingHoldoutSet$Var3 = NULL
trainingNonHoldoutSet$Var1 = NULL
trainingNonHoldoutSet$Var2 = NULL
trainingNonHoldoutSet$Var3 = NULL

trainingHoldoutSet$Var6 = NULL
trainingHoldoutSet$Var8 = NULL
trainingNonHoldoutSet$Var6 = NULL
trainingNonHoldoutSet$Var8 = NULL

gbmWithCrossValidation = gbm(formula = Response ~ ., distribution = "bernoulli", data = trainingNonHoldoutSet,
                             n.trees = 2000, shrinkage = .1, n.minobsinnode = 200, cv.folds = 5, 
                             interaction.depth = 2, n.cores = 4)
bestTreeForPrediction = gbm.perf(gbmWithCrossValidation)
gbmHoldoutPredictions = predict(object = gbmWithCrossValidation, newdata = trainingHoldoutSet, 
                                n.trees = bestTreeForPrediction, type = "response")
print(paste(LogLossBinary(train$Response[randomRows], gbmHoldoutPredictions), "Holdout Log Loss"))
# 0.580976869206406
gbmTestPredictions = predict(object = gbmWithCrossValidation, newdata = test, n.trees = 2000, type = "response")
outputDataSet = data.frame("RowID" = test$RowID, "ProbabilityOfResponse" = gbmTestPredictions)
write.csv(outputDataSet, "gbm_8.csv", row.names = FALSE)
# public leaderboard: 0.58092 (did not improve from previous score of 0.58008)


#### create dataset for neural network ####
randomSeed = 1337
set.seed(randomSeed)

train = read.csv("train.csv")
test = read.csv("test.csv")
train$ind = 'train'
test$ind = 'test'
test$Response = as.integer(0)
train$Model = as.character(train$Model)
train$Make = as.character(train$Make)
test$Model = as.character(test$Model)
test$Make = as.character(test$Make)
levels(test$Cat3) = levels(train$Cat3)
all_data = union(train, test)

all_data$Model = NULL
all_data$age_at_ins = all_data$CalendarYear - all_data$ModelYear + 1
all_data$CalendarYear = NULL
all_data$ModelYear = NULL
all_data$OrdCat = as.factor(all_data$OrdCat)

all_data$Var1 = NULL
all_data$Var2 = NULL
all_data$Var3 = NULL
all_data$Var6 = NULL
all_data$Var8 = NULL

write.csv(all_data, file = 'all_data_nn.csv', row.names=FALSE, quote = FALSE, eol = "\n")


