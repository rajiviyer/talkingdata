setwd("C:/Rajiv/Work/Data Science/Kaggle/talkingdata")
library(data.table)
library(xgboost)

crossv <- 1

pbdm <- fread("phone_brand_device_model.csv",colClasses=c("character","character","character"))
events <- fread("events.csv",colClasses=c("character","character","character","numeric","numeric"))
app_events <- fread("app_events.csv",colClasses=c("character","character","character","integer"))
app_labels <- fread("app_labels.csv",colClasses=c("character","character"))
label_categories <- fread("label_categories.csv",colClasses=c("character","character"))
gatr <- fread("gender_age_train.csv",colClasses=c("character","character","integer","character"))
gate <- fread("gender_age_test.csv",colClasses=c("character"))

pbdm <- pbdm[!duplicated(pbdm),]
pbdm <- pbdm[-which(duplicated(pbdm$device_id)),]


pbdm <- pbdm[,":="(phone_brand=paste0("brand:",as.numeric(as.factor(phone_brand))),
				   device_model=paste0("model:",as.numeric(as.factor(device_model)))
				   )]
app_labels <- unique(app_labels[,.(app_id,label_id)],by=NULL)
app_labels <- merge(app_labels,label_categories,by="label_id",all=FALSE)
app_events <- merge(events,app_events,by="event_id",all=FALSE)

#device_app_cnt <- app_events[,.(app_cnt = .N), by=.(device_id,event_id)][,.(avg_app_cnt=mean(app_cnt)),by=.(device_id)]
#device_app_active_cnt <- app_events[,.(tot_active = sum(is_active), tot_inactive = sum(ifelse(is_active,0,1))), by=.(device_id,event_id)][,.(avg_actapp_cnt = mean(tot_active),avg_inactapp_cnt = mean(tot_inactive) ),by=.(device_id)]
#device_long <- app_events[,.(long = mean(longitude)), by=.(device_id,event_id)][,.(avg_long = mean(long)),by=.(device_id)]
#device_lat <- app_events[,.(lat = mean(latitude)), by=.(device_id,event_id)][,.(avg_lat = mean(lat)),by=.(device_id)]
device_apps <- unique(app_events[,.(device_id,app_id)],by=NULL) 
device_labels <- as.data.table(sqldf("select * from app_labels a, device_apps b where a.app_id = b.app_id"))
device_labels <- unique(device_labels[,.(device_id,category)],by=NULL)
device_labels$category <- paste0("label:",as.numeric(as.factor(device_labels$category)))
device_labels <- dcast.data.table(device_labels,device_id ~ category, length, value.var="device_id", fill=0)
device_apps$app_id <- paste0("app:",as.numeric(as.factor(device_apps$app_id)))
rm(app_events,events)
gc()
tst <- as.data.frame(table(device_apps$app_id),stringsAsFactors=FALSE)
jlevels <- tst[tst$Freq<200,]$Var1
device_apps[device_apps$app_id %in% jlevels,]$app_id <- "app:Others"
device_apps <- dcast.data.table(device_apps, device_id ~ app_id, length, value.var="device_id", fill=0)

gatr <- gatr[,":="(gender = NULL, age = NULL)]
gate$group <- NA
full <- rbind(gatr,gate)

full <- merge(pbdm,full, by="device_id",all=FALSE)
dev_phone <- full[,.(device_id,phone_brand)]
dev_model <- full[,.(device_id,device_model)]
dev_phone <- dcast.data.table(dev_phone, device_id ~ phone_brand, length, value.var="device_id", fill=0)
dev_model <- dcast.data.table(dev_model, device_id ~ device_model, length, value.var="device_id", fill=0)
full <- full[,":=" (phone_brand = NULL, device_model = NULL)]
#full <- merge(full,device_app_cnt,by="device_id",all.x=TRUE)
#full <- merge(full,device_app_active_cnt,by="device_id",all.x=TRUE)
#full <- merge(full,device_long,by="device_id",all.x=TRUE)
#full <- merge(full,device_lat,by="device_id",all.x=TRUE)
full <- merge(full,dev_phone,by="device_id",all=FALSE)
full <- merge(full,dev_model,by="device_id",all=FALSE)
full <- merge(full,device_apps,by="device_id",all.x=TRUE)
full <- merge(full,device_labels,by="device_id",all.x=TRUE)

#rm(dev_phone,dev_model,device_apps,device_app_cnt,device_app_active_cnt,device_long,device_lat)
rm(dev_phone,dev_model,device_apps,device_labels)
gc()

for (col in 3:ncol(full)) set(full, which(is.na(full[[col]])), col, 0)

train <- full[!is.na(full$group)]
col_cnt <- colSums(train[,!c("device_id","group"),with=FALSE])
zero_cols <- names(col_cnt[col_cnt==0])
full <- full[,!zero_cols,with=FALSE]
train <- full[!is.na(full$group)]
train_id <- train$device_id
train_lab <- as.numeric(as.factor(train$group))-1
train <- train[,":=" (device_id = NULL, group = NULL)]
test <- full[is.na(full$group)]
test_id <- test$device_id
test <- test[,":=" (device_id = NULL, group = NULL)]

rm(full)
gc()

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
bst.cv <- xgb.cv(param=param, data = trainMatrix, label=train_lab, nfold = 5, nrounds = 500,early.stop.round=5,maximize=FALSE,verbose=T)
best <- min(bst.cv$test.mlogloss.mean)
bestIter <- which(bst.cv$test.mlogloss.mean==best)
} else
{
bestIter=100
}
xgb_mod <- xgboost(param=param, data=trainMatrix, label=train_lab, nrounds=bestIter, verbose=1,print.every.n = 5)

pred <- predict(xgb_mod,testMatrix)
pred <- t(matrix(pred,nrow=length(unique(train_lab))))
my_solution <- data.frame(test_id,pred)
colnames(my_solution) <- c("device_id","F23-","F24-26","F27-28","F29-32","F33-42","F43+","M22-","M23-26","M27-28","M29-31","M32-38","M39+")
write.csv(my_solution,file="xgb_lab.csv",row.names=FALSE)