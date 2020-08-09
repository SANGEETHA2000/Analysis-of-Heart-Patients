convert.magic <- function(obj,types) 
{
  for (i in 1:length(types)) 
  {
    if(types[i]=='factor'){
      obj[,i]=as.factor(obj[,i])
    }else{
      obj[,i]=as.numeric(obj[,i])
    }
  } 
  return(obj) 
}

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Positive', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Negative', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'True', cex=1.2, srt=90)
  text(140, 335, 'False', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
} 


#Preprocessing the Data
data <- read.csv("heart.csv")
names(data)<-c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target")
s<-sum(is.na(data))
if(!s)
  data<-na.omit(data)
chclass <-c("numeric","factor","factor","numeric","numeric","factor","factor","numeric","factor","numeric","factor","factor","factor","factor")
data <- convert.magic(data,chclass)
options(warn=-1)

#Analysis of Factors
barplot(table(data$target),main="General Analysis", col=c("grey","black"), names.arg = c("Normal", "Diseased"))

levels(data$target) = c("Disease","No Disease")
levels(data$sex) = c("Female","Male")

mosaicplot(data$sex ~ data$target,
           main="Analysis by Gender", shade=FALSE,color=TRUE,
           xlab="Gender", ylab="Heart disease")

boxplot(data$age ~ data$target,
        main="Fate by Age",
        ylab="Age",xlab="Heart disease")

levels(data$target)<-c(0,1)
d<-data$age[data$target==1]
hist(d, main="Age Distribution of Diseased Patients", breaks = "Sturges",xlab="Age",col="red" )
mean(d)
nd<-data$age[data$target==0]
hist(nd,main="Age Distribution of Normal Patients",breaks = "Sturges", xlab="Age", col="grey")

#Splitting the Data
set.seed(10)
inTrainRows <- createDataPartition(data$target,p=0.75,list=FALSE)
trainData <- data[inTrainRows,]
testData <-  data[-inTrainRows,]
nrow(trainData)/(nrow(testData)+nrow(trainData))
nrow(testData)/(nrow(testData)+nrow(trainData))

#Storage
logReg<-c(0)  #Logical Regression
boost<-c(0)  #Boosted Tree
RF<-c(0)  #Random Forest
svm<-c(0) #Support Vector Machine
AUC<-data.frame(logReg, boost, RF, svm)
Accuracy <-data.frame(logReg, boost, RF, svm)

#Logistic Regression Model

logRegModel <- train(target ~ ., data=trainData, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, testData)
logRegPredictionprob <- predict(logRegModel, testData, type='prob')[2]
logRegConfMat <- confusionMatrix(logRegPrediction, testData[,"target"])

#ROC Curve
AUC$logReg <- roc(as.numeric(testData$target),as.numeric(as.matrix(logRegPredictionprob)))$auc
Accuracy$logReg<- logRegConfMat$overall['Accuracy']

draw_confusion_matrix(logRegConfMat)
logRegImp=varImp(logRegModel, scale=FALSE)
plot(logRegImp,main = 'Variable importance with Logistic Regression')

#(1/1+e^-z)

#Random Forest Model

RFModel <- randomForest(target ~ .,
                        data=trainData,
                        importance=TRUE,
                        ntree=2000)
#varImpPlot(RFModel)
RFPrediction <- predict(RFModel, testData)
RFPredictionprob = predict(RFModel,testData,type="prob")[, 2]

RFConfMat <- confusionMatrix(RFPrediction, testData[,"target"])

AUC$RF <- roc(as.numeric(testData$target),as.numeric(as.matrix((RFPredictionprob))))$auc
Accuracy$RF <- RFConfMat$overall['Accuracy']


draw_confusion_matrix(RFConfMat)
RFImp=varImp(RFModel, scale=FALSE)
plot(svmImp,main = 'Variable importance with Random Forest')

#Boosted Model

objControl <- trainControl(method='cv', number=10)#  repeats = 10)
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode =10)
# run model
boostModel <- train(target ~ .,data=trainData, method='gbm',
                    trControl=objControl, tuneGrid = gbmGrid, verbose=F)
boostPrediction <- predict(boostModel, testData)
boostPredictionprob <- predict(boostModel, testData, type='prob')[2]
boostConfMat <- confusionMatrix(boostPrediction, testData[,"target"])

#ROC Curve
AUC$boost <- roc(as.numeric(testData$target),as.numeric(as.matrix((boostPredictionprob))))$auc
Accuracy$boost <- boostConfMat$overall['Accuracy'] 


boostImp =varImp(boostModel, scale = FALSE)
row = rownames(varImp(boostModel, scale = FALSE)$importance)

draw_confusion_matrix(boostConfMat)
rownames(boostImp$importance)=row
plot(boostImp,main = 'Variable importance with boosted tree')

#Support Vector Machine
feature.names=names(data)
for (f in feature.names) {
  if (class(data[[f]])=="factor") {
    levels <- unique(c(data[[f]]))
    data[[f]] <- factor(data[[f]],
                        labels=make.names(levels))
  }
}

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

inTrainRows <- createDataPartition(data$target,p=0.75,list=FALSE)
trainData2 <- data[inTrainRows,]
testData2 <-  data[-inTrainRows,]
svmModel <- train(target ~ ., data = trainData2,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
svmPrediction <- predict(svmModel, testData2)
svmPredictionprob <- predict(svmModel, testData2, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, testData2[,"target"])
#ROC Curve
AUC$svm <- roc(as.numeric(testData2$target),as.numeric(as.matrix((svmPredictionprob))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']

draw_confusion_matrix(svmConfMat)
svmImp=varImp(svmModel, scale=FALSE)
plot(svmImp,main = 'Variable importance with Support Vector Machine')

print("Summary of Age")
print(summary(data$age))
print("Summary of Sex")
print(summary(data$sex))
print("Summary of Chest pain Type")
print(summary(data$cp))
print("Summary of Resting bp")
print(summary(data$trestbps))
print("Summary of Cholesterol")
print(summary(data$chol))
print("Summary of Fasting Blood Sugar")
print(summary(data$fbs))
print("Summary of Resting Electrographic Results")
print(summary(data$restecg))
print("Summary of Maximum Heart Rate")
print(summary(data$thalach))
print("Summary of Exercise induced angina")
print(summary(data$exang))
print("Summary of ST depression induced by exercise relative to rest")
print(summary(data$oldpeak))
print("Summary of Slope of the peak")
print(summary(data$slope))
print("Summary of Number of major vessels colored")
print(summary(data$ca))
print("Summary of Defect type")
print(summary(data$thal))
print("Summary of Diagnosis of heart disease")
print(summary(data$target))
cat("\n")
print("Area Under the ROC Curve")
print(AUC)
print("Accuracy of Predictions")
print(Accuracy)
cat("\n")


