pacman::p_load(caret, lattice, tidyverse, gam, logistf, MASS, car, corrplot, gridExtra, ROCR, RCurl, randomForest, readr, readxl)

##For lit review, write a paper that contains an analysis on bank-related data and compare what analytical techniques they used and worked

##### Data Set ######
url1 <- "https://raw.githubusercontent.com/btj5z2/DA6813/main/BBBC-Train.xlsx"
download.file(url1, "BBBC-Train.xlsx", mode = "wb")
BBBC_train <- read_excel("BBBC-Train.xlsx")

url2 <- "https://raw.githubusercontent.com/btj5z2/DA6813/main/BBBC-Test.xlsx"
download.file(url2, "BBBC-Test.xlsx", mode = "wb")
BBBC_test <- read_excel("BBBC-Test.xlsx")


#Copy of data set to model
train = BBBC_train
test = BBBC_test

#Turning character variables into factors
fac_vars = c("Choice", "Gender")
train[fac_vars] = lapply(train[fac_vars],as.factor)
test[fac_vars] = lapply(test[fac_vars],as.factor)

##### Balanced? No.##### 
plot(train$Choice)

#Correlation Plot
combined = rbind(train, test)
corrplot::corrplot(cor(combined[,c(4:12)]), method = c("number")) #First and last purchased have pretty high correlation

#Normalize data
train = scale(train[,c(4:12)]) #normalize numeric data 
train = cbind(BBBC_train[,2:3], train) #combine factor data with normalized data and leaving out the index column ("observation")
train[fac_vars] = lapply(train[fac_vars],as.factor)
str(train)

test = scale(test[,c(4:12)]) #normalize numeric data 
test = cbind(BBBC_test[,2:3], test) #combine factor data with normalized data and leaving out the index column ("observation")
test[fac_vars] = lapply(test[fac_vars],as.factor)
str(test)

#Multi Collinearity
log.model = glm(Choice ~ . , data = train, family = binomial)
vif(log.model)

log.model2 = glm(Choice ~ . -Last_purchase , data = train, family = binomial) #Remove last_purchased
vif(log.model2)

log.model3 = glm(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Youth + P_Cook + P_DIY 
                 + P_Art , data = train, family = binomial) #Remove first_purchased
vif(log.model3)

#Logistic model 
summary(log.model3) #P_Youth not significant

log.model4 = glm(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Cook + P_DIY 
                 + P_Art , data = train, family = binomial) #Remove first_purchased
summary(log.model4)



#Predictions 
test$PredProb = predict.glm(log.model4, newdata = test, type = "response")
test$PredSur = ifelse(test$PredProb >= 0.5, 1, 0) # Create new variable converting probabilities to 1s and 0s


# "ConfusionMatrix" to get accuracy of the model prediction
caret::confusionMatrix(as.factor(test$Choice), as.factor(test$PredSur)) #Comparing observed to predicted




