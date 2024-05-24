---
  
#out---
#Title: "Using machine learning approaches to create the most significant Alzheimer's disease prediction (AD)"
#Course:"Machine Learning and Data Modelling - COM737 ( 17948 )"
#Name: "Adebomojo Moyosore"
#Student ID: "B00897178"
#Email: "adebomojo-m@ulster.ac.uk"
#Date: "06/04/2023"
  
---
#1. Loading of all relevant libraries needed.

"```{r, include=FALSE}"
options(tinytex.verbose = TRUE)

library(Boruta)
library(missForest)
library(Publish)
library(caTools)
library(ggplot2)
library(corrplot) 
library(smotefamily)
library(ROCR)
library(editrules)
library(skimr)
library(randomForest)
library(dplyr)
library(MLmetrics)
library(data.table)
library(caret)
library(rmarkdown)

#1.1 Importing all necessary files into RStudio.

neurobat_ailb <- read.csv("/Users/apple/Desktop/Machine Learning and Data Modelling/Course Work/aibl_neurobat_01-Jun-2018.csv", stringsAsFactors = FALSE)
pdxconv_ailb <- read.csv("/Users/apple/Desktop/Machine Learning and Data Modelling/Course Work/aibl_pdxconv_01-Jun-2018.csv", stringsAsFactors = FALSE)
mmse_ailb <- read.csv("/Users/apple/Desktop/Machine Learning and Data Modelling/Course Work/aibl_mmse_01-Jun-2018.csv", stringsAsFactors = FALSE)
cdr_ailb <- read.csv("/Users/apple/Desktop/Machine Learning and Data Modelling/Course Work/aibl_cdr_01-Jun-2018.csv", stringsAsFactors = FALSE)
labdata_ailb <-read.csv("/Users/apple/Desktop/Machine Learning and Data Modelling/Course Work/aibl_labdata_01-Jun-2018.csv", stringsAsFactors = FALSE)
apoeres_ailb <- read.csv("/Users/apple/Desktop/Machine Learning and Data Modelling/Course Work/aibl_apoeres_01-Jun-2018.csv", stringsAsFactors = FALSE)
medhist_ailb <- read.csv("/Users/apple/Desktop/Machine Learning and Data Modelling/Course Work/aibl_medhist_01-Jun-2018.csv", stringsAsFactors = FALSE)
ptdemog_ailb <- read.csv("/Users/apple/Desktop/Machine Learning and Data Modelling/Course Work/aibl_ptdemog_01-Jun-2018.csv", stringsAsFactors = FALSE)

#1.2 Merging the datasets with 1688 observations after combining the files .

aibl_new<-cdr_ailb%>%
  inner_join(labdata_ailb)%>%
  inner_join(mmse_ailb)%>%
  inner_join(neurobat_ailb)%>%
  inner_join(pdxconv_ailb,by=c("RID"="RID","SITEID"="SITEID","VISCODE"="VISCODE"))

#1.3 the datasets with only 868 observations are combined. During the duration of the testing period, these stay constant.

medhist<-medhist_ailb[,c(-2,-3)]
apoeres<-apoeres_ailb[,c(-2,-3)]
ptdemog<-ptdemog_ailb[,c(-2,-3)]

### Merge with the key RID only.
aibl_new<-aibl_new%>%
  left_join(medhist)%>%
  left_join(apoeres)%>%
  left_join(ptdemog,by=c("RID"="RID"))

head (aibl_new)
arrange(aibl_new, RID)

sapply(aibl_new, class) # Having knowledge of the Class 
dim(aibl_new) # Examining the dataframe's size 
names(aibl_new) # Names of the variables 

aibl_new_baseline <- aibl_new[aibl_new$VISCODE=='bl', ]
aibl_new_m18 <- aibl_new[aibl_new$VISCODE=='m18', ]
aibl_raw_set <- aibl_new_baseline # Baseline dataset assignment to aibl_raw_set.
"```"
#2. Studying the dataset's underlying information and performing data pre-processing operations to clean up the raw data.
"```{r, include=FALSE}"
names(aibl_raw_set) # Examining the column name. 
(aibl_raw_set) # Upon examining the data fram .
summary(aibl_raw_set) # Overview of the dataset's data.

aibl_raw_set <- subset(aibl_raw_set, select = -c(SITEID,VISCODE,EXAMDATE,APTESTDT,RID)) # Removing SITEID, VISCODE, EXAMDATE, APTESTDT, and RID from the data since they were simply used to connect the different files together and do not have any bearing on the intended research.
aibl_raw_set$PTDOB<-gsub("/","",as.character(aibl_raw_set$PTDOB)) # Deleting the "/" character from each PTDOB column value. 
aibl_raw_set$PTDOB <- as.numeric(aibl_raw_set$PTDOB) # To make it possible to calculate a new variable, normalise the data and force it to be of the numeric data type. 
aibl_raw_set$AGE <- (2006 - aibl_raw_set$PTDOB) # For each entry in the dataset, a fresh variable (AGE) is created. The year 2006 was chosen since the study was introduced on November 14, 2006. 
aibl_raw_set <- subset(aibl_raw_set, select = -c(PTDOB)) # PTDOB is being removed from the dataset since it is no longer necessary now that the AGE has been determined. 
aibl_raw_set$DXCURREN[aibl_raw_set$DXCURREN == 3]<- 2 # Combining MCI and AD under one heading. Hence, the variables are now 1 for Healthy Control (HC) and 2 for non-HC (Non-Healthy Control). 

setDT(aibl_raw_set) 
aibl_raw_set[,MMSCORE:=cut(MMSCORE,
                           breaks=c(0,9,20,24,30),
                           include.lowest=TRUE,
                           labels=c("1","2","3","4"))] 
aibl_raw_set[,table(MMSCORE)] # Creating categories for the data in the MMSCORE column. 
aibl_rset <- aibl_raw_set # Creating a new memory to store aibl raw .
aibl_rset <- aibl_raw_set %>% mutate_if(is.numeric, ~replace(., . == -4, NA)) # Due to lacking data, all -4 values are being set to NA. 
summary(aibl_rset) # Looking at the data summary reveals that column MH16SMOK has 312 of the most missing data. 
sum(is.na(aibl_rset)) # There are 1,255 missing values in the whole dataset. 

#2.1 visualising the dataset to examine the variables more thoroughly. 

skim(aibl_rset)
boxplot(aibl_rset, cex.axis = 0.7, col.axis = "#0000FF", col.ticks = "#0000FF", horiz = TRUE, dotplot = FALSE) # Boxplot of the characteristics for a visual representation of outliers. 


#2.2 Removing errors, special characters and outliers from the data set.

install.packages("editset")
#library(editset)
(Errs <- editset(c("AGE >=55", "AGE <= 96", "PTGENDER >= 1", "PTGENDER <= 2", "APGEN1 >= 2", "APGEN2 >= 2", "APGEN1 <= 4", "APGEN2 <= 4", "DXCURREN >=1","DXCURREN <=2"))) #Checking for data consistency.
le_Errs <- violatedEdits(Errs, aibl_rset, method = "mip") #summarize and visualize the rule violations.
summary(le_Errs) #It is observed from the summary of Edit Violations that 95.2% (821 out of 862) observations of the data has no errors with only num1 and num10 rules violated. 
plot(le_Errs) #Graphical view of violatedEdits.
plot(Errs) #Based on the plot, there is no visible inter-connectivity between the edits.
viz_err <- localizeErrors(Errs, aibl_rset, method = "mip")# Using localizeErrors to obtain a boolean mask which has TRUE for each error.
aibl_rset[viz_err$adapt] <- NA  #Replace all erroneous values with NA using (the result of) localizeErrors as a boolean mask.

is.finite(c(Inf,NA,NaN)) #Special values (Inf, NA and NaN) checks and correction.
is.special <- function(x){
  if (is.numeric(x)) !is.finite(x) else is.na(x)
} #function to define special characters.
sapply(aibl_rset, is.special)  #Special character detection in the dataset.
aibl_rset[mapply(is.special, aibl_rset)] <- NA #Applying NA to all special characters.

#2.3 Outlier detection.
glimpse(aibl_rset)
aibl_rset$MMSCORE <- as.numeric(aibl_rset$MMSCORE)
outliers_aibl <- boxplot.stats(aibl_rset, coef = 2)$out #Using the Tukey's box-and-whisker method for outlier detection.
length(outliers_aibl) # There are 3,115 outliers identified within the dataset however the outliers will be retained for purpose of this study.
outliers_aibl <- NA #Converting all detected outliers to NA.
"```"
#3. Missing data imputation method using the Random Forest machine learning approach.
"```{r}"
dim(aibl_rset) #Check number of rows and columns.
str(aibl_rset) #Checks for variable type.
aibl_rset <- aibl_rset %>% mutate(across(c(DXCURREN, MHPSYCH, MH2NEURL, MH4CARD, MH6HEPAT, MH8MUSCL, MH9ENDO, MH10GAST, MH12RENA, MH16SMOK, MH17MALI, APGEN1, APGEN2, PTGENDER, LDELTOTAL, LIMMTOTAL, MMSCORE, CDGLOBAL), factor)) #Coercing variables to their appropriate data type.

#3.1 Multiple data imputation using Random Forest method (missForest).
set.seed(123)
aibl_impute <- missForest(aibl_rset,  verbose = TRUE)
aibl_impute$OOBerror # Obtaining the final true normalized root mean squared error (NRMSE) as 0.4835484 and the proportion of falsely classified (PFC) values as 0.2660085 after 4 iterations. 

names(aibl_impute) #view the data
aibl_OOBerror <- aibl_impute$OOBerror #Assign aibl_impute$OOBerror to aibl_OOBerror.
aibl_impz <- aibl_impute$ximp #Assign aibl_impute$ximp to aibl_imp.
table(aibl_impz$DXCURREN)
barplot(table(aibl_impz$DXCURREN)) #Imbalanced class with class 1 having 609 observations and class 2 having 253 observations.
#write.csv(aibl_impz, file = "Cleaned_aibl_bl_unbalanced.csv") Exporting the cleaned data to csv. This is the dataset with all errors, outliers and special characters removed and missing values imputed.
"```"
#4. Checking for multi-collinearity among the variables to evaluate if feature selection is necessary .
"```{r}"
ab_set <- aibl_impz # aibl_impz is stored as ab_set. 
ab_set <- sapply(ab_set, as.numeric) # It is necessary to coerce non-numerical variables since correlation can only be calculated between variables of the numerical type. Converting the ab_set data into a list of numeric-typed data. 
ab_set <- as.data.frame(ab_set) # Transformation into a dataframe. 
crrltns_data <- cor(ab_set)
corrplot(crrltns_data, number.cex = .9, method = "circle", type = "full", tl.cex=0.8,tl.col = "blue") # Using the correlation plot, one can see the closely connected predictor variables.
ab_crrltns <- subset(as.data.frame.table(abs(crrltns_data)), Freq < 1 & Freq > 0.8)
ab_crrltns #Six attributes have paired relationships between them, as shown in the table. DXCURREN, CDGLOBAL, HMT40, HMT3, LDELTOTAL, and LIMMTOTAL are the six variables. The qualities that should be taken into consideration will be determined by further inspections. 
"```"
#5. Dimensionality reduction involves using the Boruta algorithm to the feature selection process to identify the key characteristics. 
"```{r}"
set.seed(123)
Boruta_ab_set <- Boruta(DXCURREN~., data = ab_set, doTrace = 2, ntree = 500)


plot(Boruta_ab_set) # The graphic displays a clear separation between the significant shadow attribute's Z score and its non-important counterparts. 
Boruta_aibl_fnl<-TentativeRoughFix(Boruta_ab_set) # In the case that certain preliminary characteristics were included in the first round, confirming those qualities. 
attStats(Boruta_aibl_fnl) #8 features are confirmed important by the algorithm which are: CDGLOBAL, HMT40, HMT102, RCT20, MMSCORE, LIMMTOTAL, LDELTOTAL and APGEN1. However, it is obsereved that LIMMTOTAL and LDELTOTAL which were found to be highly correlated from the pearson's correlation plot are among the 8 selected features. The decision to drop either of the two will be done by conducting backward elimination during model training and testing.
SFI <- c(1,7,9,12,14:16,28)# Assigning the confirmed features to selectFeatureInd.
aibl_stndrd <- ab_set[,SFI]%>% mutate(across(c(APGEN1, MMSCORE, CDGLOBAL, LIMMTOTAL, LDELTOTAL), factor)) #Coercing variables to their appropriate data type.
str(aibl_stndrd)
"```"
#6. Scaling the Numeric columns in the reduced dataset.
"```{r}"
aibl_stndrd$HMT40 <- (aibl_stndrd$HMT40-min(aibl_stndrd$HMT40))/(max(aibl_stndrd$HMT40)-min(aibl_stndrd$HMT40))
aibl_stndrd$HMT102 <- (aibl_stndrd$HMT102-min(aibl_stndrd$HMT102))/(max(aibl_stndrd$HMT102)-min(aibl_stndrd$HMT102))
aibl_stndrd$RCT20 <- (aibl_stndrd$RCT20-min(aibl_stndrd$RCT20))/(max(aibl_stndrd$RCT20)-min(aibl_stndrd$RCT20))
str(aibl_stndrd)
"```"

#7. Conducting class balancing using the SMOTE Technique.
"```{r}"
aibl_stndrd <- sapply(aibl_stndrd, as.numeric)# Coercing aibl_stndrd into a list of data with numeric type.
aibl_stndrd <- as.data.frame(aibl_stndrd) #Converting to dataframe.
aibl_smt <- SMOTE(aibl_stndrd, ab_set$DXCURREN, K = 3, dup_size = 1) #SMOTE Function.
aibl_stndrd <- bind_cols(aibl_smt$data[8], aibl_smt$data[-8]) #Binding data with ab.
str(aibl_stndrd) #Structure of SMOTE data.
names(aibl_stndrd)[9]<- "DXCURREN" #Set the label colname back to "DXCURREN".
table(aibl_stndrd$DXCURREN) #Now we have a more balanced class with class 1 at 609 and class 2 at 506 observations.
barplot(table(aibl_stndrd$DXCURREN))#Boxplot to visualize the class distribution after applying the SMOTE technique.
write.csv(aibl_stndrd,file = "~/Desktop/Machine Learning and Data Modelling/Course Work/Balanced_aibl.csv") #Balanced aibl dataset with the selected features.
"```"
#8. Splitting the dataset into the Training set and Test set
"```{r}"
install.packages('caTools')
library(caTools)
aibl_stndrd$DXCURREN <- as.factor(aibl_stndrd$DXCURREN)
set.seed(123)
split = sample.split(aibl_stndrd$DXCURREN, SplitRatio = 0.7) #Training 70% and test data 30%
training_aibl <- subset(aibl_stndrd, split == TRUE)
test_aibl <- subset(aibl_stndrd, split == FALSE)
dim(training_aibl)# 780 rows and 9 columns.
dim(test_aibl)# 335 rows and 9 columns
training_aibl$DXCURREN <- as.factor(training_aibl$DXCURREN)
test_aibl$DXCURREN <- as.factor(test_aibl$DXCURREN)

"```"
#9. Fitting model using an RBF-based kernel SVM
"```{r}"
library(e1071) # The e1071 library includes a built-in tune() function to perform cross-validation which by default is 10-fold cross validation.

set.seed(123)
tn_output <- tune(svm,DXCURREN~.,data=training_aibl,kernel="radial",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)), gamma=c(0.5,1,2,3,4))
summary(tn_output)

# Showing the best model obtained
best_model <- tn_output$best.model
summary(best_model)

# Testing the model by predicting the class labels of the test data using the best model obtained via 10-FCV.

new_prdctn=predict(best_model,test_aibl)
rslt <- table(Actual=test_aibl$DXCURREN, predict=new_prdctn)

#Confusion Matrix
library(caret)
confusionMatrix(rslt, mode = "everything") #Model accuracy is 96.12%, Recall = 96.20% and F1 Score = 96.46%

#Using the backward elimination technique to conclude on dropping either LIMMTOTAL and LDELTOTAL.

training_aiblA <- training_aibl[,-8] #Dropping LDELTOTAL from the selected training dataset.
test_aiblA <- test_aibl[,-8] #Dropping LDELTOTAL from the selected test dataset.

set.seed(123)
tn_output <- tune(svm,DXCURREN~.,data=training_aiblA,kernel="radial",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)), gamma=c(0.5,1,2,3,4))
summary(tn_output)

# Showing the best model obtained
best_model <- tn_output$best.model
summary(best_model)

# Testing the model by predicting the class labels of the test data using the best model obtained via 10-FCV.

new_prdctn=predict(best_model,test_aiblA)
rslt <- table(Actual=test_aiblA$DXCURREN, predict=new_prdctn)

#Confusion Matrix
confusionMatrix(rslt, mode = "everything") # Model accuracy is 95.82%, Recall = 96.17% and F1 Score = 96.17% is observed meaning that LIMMTOTAL is an important feature.

training_aiblB <- training_aibl[,-7] #Dropping LIMMTOTAL from the selected training dataset.
test_aiblB <- test_aibl[,-7]#Dropping LIMMTOTAL from the selected test dataset.

set.seed(123)
tn_output <- tune(svm,DXCURREN~.,data=training_aiblB,kernel="radial",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)), gamma=c(0.5,1,2,3,4))
summary(tn_output)

# Showing the best model obtained
best_model <- tn_output$best.model
summary(best_model)

# Testing the model by predicting the class labels of the test data using the best model obtained via 10-FCV.

new_prdctn=predict(best_model,test_aiblB)
rslt <- table(Actual=test_aiblB$DXCURREN, predict=new_prdctn)

#Confusion Matrix
confusionMatrix(rslt, mode = "everything") # Model accuracy is 96.12%, Recall = 96.70% and F1 Score = 96.44% which also means that LDELTOTAL is an important feature. 

training_aiblC <- training_aibl[,-7:-8] #Dropping both LIMMTOTAL and LDELTOTAL from the selected training dataset.
test_aiblC <- test_aibl[,-7:-8]#Dropping both LIMMTOTAL and LDELTOTAL from the selected training dataset.

set.seed(123)
tn_output <- tune(svm,DXCURREN~.,data=training_aiblC,kernel="radial",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)), gamma=c(0.5,1,2,3,4))
summary(tn_output)

# Showing the best model obtained
best_model <- tn_output$best.model
summary(best_model)

# Testing the model by predicting the class labels of the test data using the best model obtained via 10-FCV.

new_prdctn=predict(best_model,test_aiblC)
rslt <- table(Actual=test_aiblC$DXCURREN, predict=new_prdctn)

#Confusion Matrix
confusionMatrix(rslt, mode = "everything") # Model accuracy =95.82%, recall = 96.69% and F1 Score = 96.15% which suggests that both or either of LIMMTOTAL and LDELTOTAL are important to predicting the target class.
"```"
#10.Fitting a KNN Classifier
"```{r}"
set.seed(123)
library (class)
instances <- sqrt(nrow(training_aibl))
new_instances <- round(instances, digits=0)
# Train the knn classifier with k = new_instances
valid_prdctn <- knn(train = training_aibl[,1:8], test = test_aibl[,1:8], cl = training_aibl[,9], k=new_instances)
mst_nd <- table(test_aibl$DXCURREN, valid_prdctn) 
confusionMatrix(mst_nd, mode = "everything") #Computed accuracy is 92.54%, recall = 92.93% and F1 Score = 93.19%
"```"
#11. Fitting model using Random Forest
"```{r, include=FALSE}"
library(randomForest)
set.seed(123) 
names (training_aibl)
FMURF <- tuneRF(training_aibl[-9],training_aibl$DXCURREN, ntreeTry=500,
                stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best_model <- FMURF[FMURF[, 2] == min(FMURF[, 2]), 1] #Computing the best FMURF.

print(FMURF)
print(best_model)

#Apply random forest (RF) with the optimal value of FMURF.

install.packages("randomForest")
library(randomForest)
set.seed(123)
rndmfrst <-randomForest(DXCURREN~.,data=training_aibl, FMURF=best_model, importance=TRUE,ntree=500) #Training the RF.
print(rndmfrst)
plot(rndmfrst)


new_prdctn <- predict(rndmfrst, newdata = test_aibl) #Predicting the Test set results.

library(dplyr)
library(caTools)
library(caret)
cm <- ConfusionMatrix(new_prdctn, test_aibl$DXCURREN) # Making the Confusion Matrix
(Classification.Accuracy <- 100*Accuracy(new_prdctn, test_aibl$DXCURREN))# Model Accuracy is 96.42%.
Mdl_accr <- table(test_aibl$DXCURREN, new_prdctn) 
confusionMatrix(Mdl_accr, mode = "everything")#Computed accuracy is 97.61%, recall = 98.88% and F1 Score = 97.79%

#Predict and Calculate Performance Metrics.
prdctnA <-predict(rndmfrst,newdata = test_aibl, type = "prob")

library(ROCR)
prfct_prdctn <- prediction(prdctnA[,2], test_aibl$DXCURREN)

#12. Accuracy.
(accrcy = performance(prfct_prdctn, "acc"))
plot(accrcy,main="Accuracy Curve for Random Forest",col=2,lwd=2)

# 1. Area under curve
auc <- performance(prfct_prdctn, "auc")
auc@y.values[[1]]

# 2. True Positive and Negative Rate
prdctnB <- performance(prfct_prdctn, "tpr","fpr")
# 3. Plot the ROC curve
plot(prdctnB,main="ROC Curve for Random Forest",col=3,lwd=3)
abline(a=0,b=1,lwd=2,lty=3,col="black")

rmarkdown::render("~/Desktop/Machine Learning and Data Modelling/Course Work/Moyosore_Adebomojo_B00897178_COM737( 32154 ).R")
rmarkdown::render("~/Desktop/Machine Learning and Data Modelling/Course Work/Moyosore_Adebomojo_B00897178_COM737( 32154 ).R", "pdf_document")
"```"

