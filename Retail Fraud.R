install.packages("DataExplorer")
install.packages('caTools')
install.packages('tidyverse')
install.packages('ROCR')
install.packages('DataExplorer')
install.packages('DMwR')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('party')
install.packages('naivebayes')

library(naivebayes)
library(rpart)
library(rpart.plot)
library(party)
library(DataExplorer)
library(caTools)
library(caret)
library(ggplot2)
library(ROCR)
library(ROSE)
library(plyr)
library(gbm)
library(DMwR)
library(glmnet)

train_data = read.csv('train.csv', header = TRUE, sep = '|')
data = read.csv('test.csv', header = TRUE, sep = '|')
realclass = read.csv('DMC-2019-realclass.csv', header =TRUE)
test_data = cbind(data,realclass)

# EDA
plot_missing(train_data)
sum (is.na(train_data))
colSums(sapply(train_data,is.na))
summary(train_data)
plot_histogram(train_data)
plot_density(train_data)

# Find outliers
boxplot(train_data, col = 'yellow')
boxplot(train_data$trustLevel, col = "yellow")
boxplot(train_data$totalScanTimeInSeconds, col = "yellow")
boxplot(train_data$grandTotal, col = "yellow")
boxplot(train_data$lineItemVoids, col = "yellow")
boxplot(train_data$scansWithoutRegistration, col = "yellow")
boxplot(train_data$quantityModifications, col = "yellow")
boxplot(train_data$scannedLineItemsPerSecond, col = "yellow")   # outlier
boxplot(train_data$valuePerSecond, col = "yellow")              # outlier
boxplot(train_data$lineItemVoidsPerPosition, col = "yellow")    # outlier

# Min-Max Normalization for Outliers
train_data$scannedLineItemsPerSecond = 
(train_data$scannedLineItemsPerSecond - min(train_data$scannedLineItemsPerSecond)) /
(max(train_data$scannedLineItemsPerSecond) - min(train_data$scannedLineItemsPerSecond))

train_data$valuePerSecond = 
(train_data$valuePerSecond - min(train_data$valuePerSecond)) /
(max(train_data$valuePerSecond) - min(train_data$valuePerSecond))

train_data$lineItemVoidsPerPosition = 
(train_data$lineItemVoidsPerPosition - min(train_data$lineItemVoidsPerPosition)) /
(max(train_data$lineItemVoidsPerPosition) - min(train_data$lineItemVoidsPerPosition))

prop.table(table(train_data$fraud))

# under & over sampling
train_balanced = ovun.sample(fraud ~ ., data = train_data, method = 'over')$data

# rose
train_balanced = ROSE(fraud ~ ., data = train_data)$data

prop.table(table(train_balanced$fraud))

### LOGISTIC REGRESSION ###
# logistic regression model
classifier = glm(fraud ~., train_balanced, family = binomial)

# logistic regression model with regularisation
classifier = cv.glmnet(as.matrix(train_balanced[,-10]),
                       as.matrix(train_balanced[,10]), alpha=1)

summary(classifier)

# Predicting the test data results logistic regression
prob_pred = predict(classifier, type = 'response', test_data[ ,-10] )

# Predicting the test data results logistic with regularisation
prob_pred = predict(classifier, type = 'response', as.matrix(test_data[ ,-10]) )

y_pred = ifelse(prob_pred > 0.5, 1, 0)

# logistic regression evaluation
cm = table(test_data$fraud, y_pred)
cm

sensitivity(cm)
specificity(cm)

accuracy = sum(diag(cm))/sum(cm)
accuracy

pred = prediction(y_pred, test_data$fraud)
auc = as.numeric(performance(pred, "auc")@y.values)
(auc = round(auc, 3))

perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "(1 - Specificity)",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))


### DECISION TREE ###
# decision tree classifier with gini index
tree = rpart(fraud ~ ., data=train_balanced, method="class")

# decision tree with entropy
tree = rpart(fraud ~ ., data=train_balanced, method="class", 
             parms=list(split="information"))

# decision tree with parameters
tree = rpart(fraud ~ ., data=train_balanced, method="class", 
             minsplit = 1, minbucket = 10, cp = -1)

print(tree)
rpart.plot(tree, extra = 104, nn = TRUE)
plotcp(tree)

tree_pred = predict(tree, test_data, type = "class")

cm = table(tree_pred, test_data$fraud)
cm

sensitivity(cm)
specificity(cm)

accuracy = sum(diag(cm))/sum(cm)
accuracy

PredictROC = predict(tree, test_data)
pred = prediction(PredictROC[,2], test_data$fraud)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "1 - Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))

auc = as.numeric(performance(pred, "auc")@y.values)
(auc = round(auc, 3))


### NAIVE BAYES
train_balanced$fraud = factor(train_balanced$fraud)

classifier = naive_bayes(x = train_balanced[ ,-10], y = train_balanced$fraud)

classifier = naive_bayes(x = train_balanced[ ,-10], y = train_balanced$fraud, 
                         laplace = 1)

classifier = naive_bayes(x = train_balanced[ ,-10], y = train_balanced$fraud, 
                         usekernel = TRUE )

y_pred = predict(classifier, newdata = test_data[ ,-10])

cm = table(test_data$fraud, y_pred)
cm

sensitivity(cm)
specificity(cm)

accuracy = sum((diag(cm))/sum(cm))
accuracy

pred = prediction(as.numeric(y_pred), test_data$fraud)
auc = as.numeric(performance(pred, "auc")@y.values)
(auc = round(auc, 3))

perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "(1 - Specificity)",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))



