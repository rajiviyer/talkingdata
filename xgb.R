setwd("C:/Rajiv/Work/Data Science/Kaggle/talkingdata")
library(readr)
library(dplyr)
library(lubridate)
library(caret)
library(xgboost)
library(sqldf)
library(caTools)

holdout <- 0
crossv <- 1

pbdm <- read_csv("phone_brand_device_model.csv",col_types=list(col_character(),col_character(),col_character()))
events <- read_csv("events.csv",col_types=list(col_integer(),col_character(),col_datetime(),col_number(),col_number()))
app_events <- read_csv("app_events.csv",col_types=list(col_integer(),col_number(),col_integer(),col_integer()))
gat <- read_csv("gender_age_train.csv",col_types=list(col_character(),col_character(),col_integer(),col_character()))
gate <- read_csv("gender_age_test.csv",col_type=list(col_character()))

pbdm <- pbdm[!duplicated(pbdm),]
pbdm <- pbdm[-which(duplicated(pbdm$device_id)),]
#pbdm <- mutate(pbdm,phone_brand = paste0("brand:",as.numeric(as.factor(phone_brand))),device_model=paste0("model:",as.numeric(as.factor(device_model))))
pbdm <- mutate(pbdm,phone_brand = as.numeric(as.factor(phone_brand)),device_model=as.numeric(as.factor(device_model)))

##########For cv testing#################
if(holdout) {
split <- sample.split(gat$gender,SplitRatio=0.85)
gat <- gat[split==TRUE,]
gate <- gat[split==FALSE,]
}

train <- inner_join(pbdm,gat, by="device_id")
test <- inner_join(pbdm,gate, by="device_id")

if(holdout) {
test_lab <- as.numeric(as.factor(test$group))-1
}

train_id <- train$device_id
train_lab <- as.numeric(as.factor(train$group))-1
train <- select(train,c(phone_brand,device_model))
test_id <- test$device_id
test <- select(test,c(phone_brand,device_model))

trainMatrix <- as.matrix(train)
testMatrix <- as.matrix(test)

set.seed(1424)
param <- list(booster="gblinear",
              num_class=length(unique(train_lab)),
              objective="multi:softprob",
              eval_metric="mlogloss",
              eta=0.01,
              lambda=5,
              lambda_bias=0,
              alpha=2)
if (crossv) {
bst.cv <- xgb.cv(param=param, data = trainMatrix, label=train_lab, nfold = 5, nrounds = 500,early.stop.round=10,maximize=FALSE,verbose=T)
best <- min(bst.cv$test.mlogloss.mean)
bestIter <- which(bst.cv$test.mlogloss.mean==best)
}
else
{
bestIter=500
}
xgb_mod <- xgboost(param=param, data=trainMatrix, label=train_lab, nrounds=bestIter, verbose=1,print.every.n = 5)

if(holdout) {
pred <- predict(xgb_mod,testMatrix)
pred <- t(matrix(pred,nrow=length(unique(train_lab))))
table(pred,test_lab)
}

pred <- predict(xgb_mod,testMatrix)
pred <- t(matrix(pred,nrow=length(unique(train_lab))))
my_solution <- data.frame(test_id,pred)
colnames(my_solution) <- c("device_id","F23-","F24-26","F27-28","F29-32","F33-42","F43+","M22-","M23-26","M27-28","M29-31","M32-38","M39+")
write.csv(my_solution,file="xgb.csv",row.names=FALSE)
