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

gbmModel = gbm(formula = Response ~ Var1 + Var2 + Cat5 + Cat6,
               distribution = "bernoulli",
               data = train,
               n.trees = 2500,
               shrinkage = .01,
               n.minobsinnode = 20)

gbmTrainPredictions = predict(object = gbmModel,
                              newdata = train,
                              n.trees = 1500,
                              type = "response")

head(data.frame("Actual" = train$Response, 
                "PredictedProbability" = gbmTrainPredictions))

LogLossBinary(train$Response, gbmTrainPredictions)


dataSubsetProportion = .2
randomRows = sample(1:nrow(train), floor(nrow(train) * dataSubsetProportion))
trainingHoldoutSet = train[randomRows, ]
trainingNonHoldoutSet = train[!(1:nrow(train) %in% randomRows), ]

trainingHoldoutSet$RowID = NULL
trainingNonHoldoutSet$RowID = NULL
trainingHoldoutSet$Model = NULL
trainingNonHoldoutSet$Model = NULL

gbmForTesting = gbm(formula = Response ~ Var1 + Var2 + Var3 + NVCat + NVVar1,
                    distribution = "bernoulli",
                    data = trainingNonHoldoutSet,
                    n.trees = 1500,
                    shrinkage = .01,
                    n.minobsinnode = 50)

summary(gbmForTesting, plot = FALSE)


gbmHoldoutPredictions = predict(object = gbmForTesting,
                                newdata = trainingHoldoutSet,
                                n.trees = 100,
                                type = "response")

gbmNonHoldoutPredictions = predict(object = gbmForTesting,
                                   newdata = trainingNonHoldoutSet,
                                   n.trees = 100,
                                   type = "response")

print(paste(LogLossBinary(train$Response[randomRows], gbmHoldoutPredictions), 
            "Holdout Log Loss"))
print(paste(LogLossBinary(train$Response[!(1:nrow(train) %in% randomRows)], gbmNonHoldoutPredictions), 
            "Non-Holdout Log Loss"))

gbmWithCrossValidation = gbm(formula = Response ~ .,
                             distribution = "bernoulli",
                             data = trainingNonHoldoutSet,
                             n.trees = 2000,
                             shrinkage = .1,
                             n.minobsinnode = 200, 
                             cv.folds = 5,
                             n.cores = 1)

bestTreeForPrediction = gbm.perf(gbmWithCrossValidation)

gbmHoldoutPredictions = predict(object = gbmWithCrossValidation,
                                newdata = trainingHoldoutSet,
                                n.trees = bestTreeForPrediction,
                                type = "response")

gbmNonHoldoutPredictions = predict(object = gbmWithCrossValidation,
                                   newdata = trainingNonHoldoutSet,
                                   n.trees = bestTreeForPrediction,
                                   type = "response")

print(paste(LogLossBinary(train$Response[randomRows], gbmHoldoutPredictions), 
            "Holdout Log Loss"))
print(paste(LogLossBinary(train$Response[!(1:nrow(train) %in% randomRows)], gbmNonHoldoutPredictions), 
            "Non-Holdout Log Loss"))

gbmForTesting = gbm(formula = Response ~ Var1 + Var2 + Var3 + NVCat + NVVar1 + NVVar2,
                    distribution = "bernoulli",
                    data = trainingNonHoldoutSet,
                    n.trees = 1500,
                    shrinkage = .1,
                    n.minobsinnode = 50)

gbmTestPredictions = predict(object = gbmWithCrossValidation,
                             newdata = test,
                             n.trees = 1000,
                             type = "response")

outputDataSet = data.frame("RowID" = test$RowID,
                           "ProbabilityOfResponse" = gbmTestPredictions)

write.csv(outputDataSet, "gbmBenchmarkSubmission.csv", row.names = FALSE)

