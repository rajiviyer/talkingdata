setwd("C:/Rajiv/Work/Data Science/Kaggle/talkingdata")
library(data.table)
library(xgboost)

holdout <- 0
crossv <- 1

pbdm <- fread("phone_brand_device_model.csv",colClasses=c("character","character","character"))
events <- fread("events.csv",colClasses=c("character","character","character","numeric","numeric"))
app_events <- fread("app_events.csv",colClasses=c("character","character","character","character"))
gatr <- fread("gender_age_train.csv",colClasses=c("character","character","integer","character"))
gate <- fread("gender_age_test.csv",colClasses=c("character"))

pbdm <- pbdm[!duplicated(pbdm),]
pbdm <- pbdm[-which(duplicated(pbdm$device_id)),]

#pbdm <- pbdm[,":="(phone_brand=as.numeric(as.factor(phone_brand)),
#				   device_model=as.numeric(as.factor(device_model))
#				   )]

pbdm <- pbdm[,":="(phone_brand=paste0("brand:",as.numeric(as.factor(phone_brand))),
				   device_model=paste0("model:",as.numeric(as.factor(device_model)))
				   )]

app_events <- merge(events,app_events,by="event_id",all=FALSE)
device_apps <- unique(app_events[,.(device_id,app_id)],by=NULL) 
device_apps$app_id <- paste0("app:",as.numeric(as.factor(device_apps$app_id)))
rm(app_events,events)
gc()
tst <- as.data.frame(table(device_apps$app_id),stringsAsFactors=FALSE)
jlevels <- tst[tst$Freq<600,]$Var1
device_apps[device_apps$app_id %in% jlevels,]$app_id <- "app:Others"
device_apps <- dcast.data.table(device_apps, device_id ~ app_id, length, value.var="device_id", fill=0)

if(holdout) {
split <- sample.split(gat$gender,SplitRatio=0.85)
gatr <- gat[split==TRUE,]
gate <- gat[split==FALSE,]
}

gatr$age <- NULL
gatr$gender <- NULL
gate$group <- NA
full <- rbind(gatr,gate)

full <- merge(pbdm,full, by="device_id",all=FALSE)
dev_phone <- full[,.(device_id,phone_brand)]
dev_model <- full[,.(device_id,device_model)]
dev_phone <- dcast.data.table(dev_phone, device_id ~ phone_brand, length, value.var="device_id", fill=0)
dev_model <- dcast.data.table(dev_model, device_id ~ device_model, length, value.var="device_id", fill=0)
full <- full[,":=" (phone_brand = NULL, device_model = NULL)]
full <- merge(full,dev_phone,by="device_id",all=FALSE)
full <- merge(full,dev_model,by="device_id",all=FALSE)
full <- merge(full,device_apps,by="device_id",all.x=TRUE)
rm(dev_phone,dev_model,device_apps)
gc()
#num_cols <- names(full)[5:ncol(full)]
for (col in 3:ncol(full)) set(full, which(is.na(full[[col]])), col, 0)

train <- full[!is.na(full$group)]
train_id <- train$device_id
train_lab <- as.numeric(as.factor(train$group))-1
train <- train[,":=" (device_id = NULL, group = NULL)]
test <- full[is.na(full$group)]
test_id <- test$device_id
test <- test[,":=" (device_id = NULL, group = NULL)]

train <- train[, lapply(.SD, as.numeric)]
test <- test[, lapply(.SD, as.numeric)]

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
bst.cv <- xgb.cv(param=param, data = trainMatrix, label=train_lab, nfold = 5, nrounds = 400,early.stop.round=10,maximize=FALSE,verbose=T)
best <- min(bst.cv$test.mlogloss.mean)
bestIter <- which(bst.cv$test.mlogloss.mean==best)
}
else
{
bestIter=360
}
xgb_mod <- xgboost(param=param, data=trainMatrix, label=train_lab, nrounds=bestIter, verbose=1,print.every.n = 5)

if(holdout) {
pred <- predict(xgb_mod,testMatrix)
table(pred>0.5,test_lab)
}

pred <- predict(xgb_mod,testMatrix)
pred <- t(matrix(pred,nrow=length(unique(train_lab))))
my_solution <- data.frame(test_id,pred)
colnames(my_solution) <- c("device_id","F23-","F24-26","F27-28","F29-32","F33-42","F43+","M22-","M23-26","M27-28","M29-31","M32-38","M39+")
write.csv(my_solution,file="xgb_dc.csv",row.names=FALSE)