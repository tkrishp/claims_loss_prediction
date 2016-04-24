setwd("~/github/claims_loss_prediction")

train = read.csv("train.csv")
test = read.csv("test.csv")

library(corrplot)
library(tidr)
library(dplyr)

# determine correlation between var variables
train_subset = select(train %>% contains("Var"))
corrplot.mixed(cor(train_subset))
# following variables have high correlation
# Var1, Var3
# Var1, Var5
# Var1, Var6
# Var2, Var4
# Var3, Var5
# Var3, Var6
# Remove variables: Var1, Var2, Var3

boxplot(train$Var4)
boxplot(train$Var5)
boxplot(train$Var7)
boxplot(train$NVVar1)
boxplot(train$NVVar2)
boxplot(train$NVVar3)
boxplot(train$NVVar4)



