setwd("~/github/claims_loss_prediction")

# import libraries
library(dplyr)
library(tidyr)
library(robustHD)
library(randomForest)

#### log loss function ####
LogLossBinary = function(actual, predicted, eps = 1e-15) {  
  predicted = pmin(pmax(predicted, eps), 1-eps)  
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

#### load data ####
train = read.csv('train.csv')
test = read.csv('test.csv')

train$ind = 'train'
test$ind = 'test'
test$Response = 0

all_data = bind_rows(train, test)

# transform all data
# add 1 to ensure age is >= 0
all_data$age_at_ins = all_data$CalendarYear - all_data$ModelYear + 1
all_data$age_at_ins = winsorize(all_data$age_at_ins, standardized = FALSE)
all_data$age_at_ins = (all_data$age_at_ins - min(all_data$age_at_ins))/(max(all_data$age_at_ins) - min(all_data$age_at_ins))

all_data$Cat1 = as.factor(ifelse(all_data$Cat1 == 'B', 'B', 'U'))
all_data$Cat2 = as.factor(ifelse(all_data$Cat2 == 'C', 'C', 'U'))
all_data$Cat3 = as.factor(ifelse(all_data$Cat3 == 'B', 'B', 
                                   ifelse(all_data$Cat3 == 'D', 'D', 
                                          ifelse(all_data$Cat3 == 'F', 'F', 'U'))))
all_data$Cat8 = as.factor(ifelse(all_data$Cat8 == 'B', 'B', 'U'))
all_data$Cat9 = as.factor(ifelse(all_data$Cat9 == 'B', 'B', 'U'))
all_data$Cat11 = as.factor(ifelse(all_data$Cat11 == 'B', 'B', 'U'))
all_data$OrdCat = as.factor(ifelse(all_data$OrdCat == 4, '4', 'U'))

all_data$ModelYear = as.factor(all_data$ModelYear)
all_data$OrdCat = as.factor(all_data$OrdCat)

train.data = all_data %>% filter(ind == 'train') %>% select(-RowID, -CalendarYear, -Make, -Model, -ind)
test.data = all_data %>% filter(ind == 'test') %>% select(-RowID, -CalendarYear, -Make, -Model, -Response, -ind)


set.seed(1234)

#### logistic regression ####
lr.model = glm(Response ~ ModelYear + Cat1 + Cat2 + Cat3 + Cat4 + Cat5 + Cat6 + Cat7 + Cat8 + Cat9 + Cat10 + Cat11 + Cat12 + OrdCat + Var1 + Var2 + Var3 + Var4 + Var5 + Var6 + Var7 + Var8 + NVCat + NVVar1 + NVVar2 + NVVar3 + NVVar4, 
               data=train.data, 
               family=binomial(link = 'logit'))
summary(lr.model)

# signficant variables: 
# ModelYear
# Cat1B, Cat2C, Cat3B, Cat3D, Cat3F, Cat8B, Cat9B, Cat11B, 
# OrdCat4, 
# Var3, Var5, Var8, 
# NVCatE, NVCatF, NVCatH, NVCatJ, NVCatM, NVCatN, NVCatO, 
# NVVar1, NVVar2, NVVar3, NVVar4
train.data = select(train.data, -Cat4, -Cat5, -Cat6, -Cat7, -Cat10, -Cat12, -Var1, -Var2, -Var4, -Var6, -Var7)
test.data = select(test.data, -Cat4, -Cat5, -Cat6, -Cat7, -Cat10, -Cat12, -Var1, -Var2, -Var4, -Var6, -Var7)

response = as.factor(train.data$Response)
model.rf = randomForest(train.data[,-17], response, ntree=100, mtry=4, importance=TRUE)
varImpPlot(model.rf)
pred = predict(model.rf, test.data, type = 'prob')

submit = read.csv("submissionExample.csv")
submit[,2] = pred[,2]
write.csv(submit,"submit_rf.csv",row.names=FALSE)


