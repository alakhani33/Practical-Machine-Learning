
# Machine Learning Algorithm to Predict How Well People Exercise

Author: Murtuza Ali Lakhani
Date: October 25, 2015

## Executive Summary

In this project, we analyzed a large volume of data from accelerometers on the belt, forearm, arm, and dumbell of six participants, who were asked to perform barbell lifts correctly and incorrectly in 5 different ways, in order to develop a cross-validated machine learning model that would make predictions about how well people exercise.  To this end, we split the given training data into training and testing sets, tested various machine learning algorithms, and validated the prediction results using a confusion matrix.  Random Forest was selected as the final model, whose prediction results were further cross-validated against the given testing dataset.        

## 1. Secure clean data
In this step, we load up the requisite libraries and data sets (Groupware@LES, 2015).

### 1.1. Onboard the requisite libraries, only they don't already exist

```r
if (!("caret" %in% rownames(installed.packages())) ) {
  install.packages("caret")
} else {
  library(caret)
}

if (!("randomForest" %in% rownames(installed.packages())) ) {
  install.packages("randomForest")
} else {
  library(randomForest)
}

if (!("e1071" %in% rownames(installed.packages())) ) {
  install.packages("e1071")
} else {
  library(e1071)
}

if (!("ggplot2" %in% rownames(installed.packages())) ) {
  install.packages("ggplot2")
} else {
  library(ggplot2)
}

if (!("Hmisc" %in% rownames(installed.packages())) ) {
  install.packages("Hmisc")
} else {
  library(Hmisc)
}
```

#### 1.2. Load the data
Loading the given training and testing data sets into the structure of the project area.

```r
setwd("C:/Users/murlakha/Desktop/DATA SCIENCE/PRACTICAL MACHINE LEARNING/PROJECT")

url_train ="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# file names
name_train = "./data/pml-training.csv"
name_test = "./data/pml-testing.csv"
# if directory does not exist, create new
if (!file.exists("./data")) {
  dir.create("./data")
}
# if files does not exist, download the files
if (!file.exists(name_train)) {
  download.file(url_train, destfile=name_train)
}
if (!file.exists(name_test)) {
  download.file(url_test, destfile=name_test)
}

# load the CSV files
train_data = read.csv("./data/pml-training.csv")
test_data = read.csv("./data/pml-testing.csv")
#dim(train_data)
#names(train_data)
```

#### 1.3. Clean the data
The first column X is not meaningful to the prediction model and is therefore dropped.  Next any columns with blank values and NAs are removed.

```r
train_data <- subset (train_data , select = -X: -num_window)
test_data <- subset (test_data , select = -X: -num_window)

train_data <- train_data[, sapply(train_data, function(x) !any(is.na(x)))]
train_data <- train_data[, sapply(train_data, function(x) !any(x==""))]
dim(train_data)
```

```
## [1] 19622    53
```

```r
test_data <- test_data[, sapply(test_data, function(x) !any(is.na(x)))]
#dim(test_data)
```

## 2. Partition the training data into two sets
In this step, the given training data is partitioned and rearranged into two sets for training and validation purposes with a seed value for reproducibility.  The dependent variable, classe, is used as the hinge for data splitting.  75% of data is allotted to training and 25% to testing.

```r
set.seed (54321)
train_partition_index = createDataPartition(train_data$classe, p = 3/4)[[1]]
training_partition_data = train_data[train_partition_index,]
testing_partition_data = train_data[-train_partition_index,]
```

## 3. Fit the model
In this step, we try the randomforest algorithm to find a fit.

```r
independent_variables <- training_partition_data [, -53]
dependent_variable <- training_partition_data [, 53]
randomforest_model <- randomForest (independent_variables, dependent_variable)
```

## 4. Validate the model--and Measure Out-of-Sample Error
In this step, we validate the model using the "test" partition of the training data

```r
predClasse <- predict (randomforest_model, testing_partition_data)
confusionMatrix (testing_partition_data$classe,predClasse)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    5  944    0    0    0
##          C    0    7  847    1    0
##          D    0    0    5  799    0
##          E    0    0    1    4  896
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9953        
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.2855        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9941        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9926   0.9930   0.9938   1.0000
## Specificity            1.0000   0.9987   0.9980   0.9988   0.9988
## Pos Pred Value         1.0000   0.9947   0.9906   0.9938   0.9945
## Neg Pred Value         0.9986   0.9982   0.9985   0.9988   1.0000
## Prevalence             0.2855   0.1939   0.1739   0.1639   0.1827
## Detection Rate         0.2845   0.1925   0.1727   0.1629   0.1827
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9982   0.9957   0.9955   0.9963   0.9994
```
Based on the confusion matrix output, the out-of-sample error rates for the five classes are as follows:

Out-of-Sample error for A = (100% - 100%(1395/1395)) = 0.0%
Out-of-Sample error for B = (100% - 100%(944/949)) = 0.5%
Out-of-Sample error for C = (100% - 100%(847/854)) = 0.8%
Out-of-Sample error for D = (100% - 100%(799/804)) = 0.6%
Out-of-Sample error for E = (100% - 100%(894/900)) = 0.7%

The out-of-sample errors can be balanced by setting different weights for the classes.  The higher the weight a class is given, the more its error rate is decreased (Stanford, 2015).

## 5. Cross-validate the model on the given test data

```r
PredictTest20 <- predict (randomforest_model, test_data)
PredictTest20
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

##6. Conclusion
Random Forest was selected as the final model for predicting how well people exercise based on data on a number of measured variables.  The overall accuracy of the model was 99.53%, with a confidence interval of 99.3% and 99.7%.  The maximum out-of-sample error for any class of the predicted variable is less than 1%.  The p-value for the overall prediction model was 2.2e-16, which indicated that the model was strongly significant.

******
# References

Groupware@LES, 2005.  Retrieved from  http://groupware.les.inf.puc-rio.br/har, on October 25, 2015.

Stanford, 2015.  Retrieved from https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#balance, on October 25, 2015.


******
# Appendix: Creation of files for submission of assignment

```r
path = "./answer"
answers=PredictTest20
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```

