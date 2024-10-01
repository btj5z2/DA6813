pacman::p_load(caret, lattice, tidyverse, gam, logistf, MASS, car, corrplot, gridExtra, ROCR, RCurl, randomForest)

##For lit review, write a paper that contains an analysis on bank-related data and compare what analytical techniques they used and worked

##### Data Set ######

#bank = read.csv("E:/UTSA/DA6813 Data Analytics Applications/Case Study 1/bank-additional.csv", sep = ";")
bank = as.data.frame(read.csv(text = getURL('https://raw.githubusercontent.com/btj5z2/DA6813/main/bank-additional.csv'), sep = ';'))

str(bank)

#Turning character variables into factors
fac_vars = c("job", "marital", "education", "default", "housing", "loan", "contact", 
             "month", "day_of_week", "poutcome", "y")
bank[fac_vars] = lapply(bank[fac_vars],as.factor)


#Balanced? No.
plot(bank$y)

#visualization

# boxplots for numeric variables

box_age = ggplot(bank, aes(y, age)) +
  geom_boxplot()

box_duration = ggplot(bank, aes(y, duration)) +
  geom_boxplot()

box_campaign = ggplot(bank, aes(y, campaign)) +
  geom_boxplot()

box_pdays = ggplot(bank, aes(y, pdays)) +
  geom_boxplot()

box_previous = ggplot(bank, aes(y, previous)) +
  geom_boxplot()

grid.arrange(box_age, box_duration, box_campaign, box_pdays, box_previous,
             ncol = 4)

# side-by-side bar plots for categorical variables

p = ggplot(bank) +
  facet_wrap(~ y) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

p + geom_bar(aes(x = job))
p + geom_bar(aes(x = marital))
p + geom_bar(aes(x = education))
p + geom_bar(aes(x = default))
p + geom_bar(aes(x = housing))
p + geom_bar(aes(x = loan))
p + geom_bar(aes(x = contact))
p + geom_bar(aes(x = month))
p + geom_bar(aes(x = day_of_week))
p + geom_bar(aes(x = poutcome))

#Multicollinearity
vif(lm(bank[,c(1,11:13,15:19)])) #3 numeric columns with high VIF (i.e. >10) : emp.var.rate, euribor3m, and nr.employed


#Missing values
lapply(bank,unique)
#unknowns in job, marital, education, default, housing, loan, 
miss = subset(bank, bank$job == "unknown")#39 observations 
miss = subset(bank, bank$marital == "unknown")#11 observations 
miss = subset(bank, bank$education == "unknown")#167 observations 
miss = subset(bank, bank$default == "unknown")#803 observations 
miss = subset(bank, bank$housing == "unknown")#105 observations 
miss = subset(bank, bank$loan == "unknown")#105 observations 

#Clean 
#Remove duration because when duration=0, non-contacted and therefore, perfectly correlated with y
bank = bank[,!(names(bank) %in% "duration")]
#Remove single illiterate education observation because if model not trained with, gives error when predicting on test data
bank = subset(bank, !(bank$education == "illiterate")) #removed 1 observation
#pdays=999 means not previously contacted
#Create new factor variable for pdays ("recently contacted, "not contacted" etc.) 
bank$pdaysdummy = ifelse(bank$pdays == 999, "Not contacted", 
                         ifelse(bank$pdays <= 7, "1 Week",
                                ifelse(bank$pdays <= 14, "2 Weeks", 
                                       ifelse(bank$pdays <= 21, "3 Weeks", "3+ Weeks"))))
plot(as.factor(bank$pdaysdummy))
bank$pdaysdummy  = as.factor(bank$pdaysdummy)


#Train/Test Split
set.seed(1)
train_partition = createDataPartition(bank$y, p = 0.8)[[1]]
train  = bank[train_partition,]
test   = bank[-train_partition,]

#Logistic model with all variables
a1 = glm(y ~ . -pdays, data = train, family = binomial)
summary(a1)

#Predictions
test$PredProb = predict.glm(a1, newdata = test, type = "response")
#Convert probabilities to 1s and 0s
test$PredSur = ifelse(test$PredProb >= 0.85, "yes", "no") #Adjusted prob to increase specificity 

# Finally, we will use the command "confusionMatrix" from the package caret to get accuracy of the model prediction. 
caret::confusionMatrix(as.factor(test$y), as.factor(test$PredSur)) #Comparing observed to predicted
 

### Random Forest Model ###
set.seed(1)
rf_model = train(y ~ . -pdays, data = train, method = "rf", trControl = trainControl(method = "cv", number = 10))
print(rf_model)

# Predictions for random forest
test$RF_Pred = predict(rf_model, newdata = test)

# Confusion matrix for random forest
caret::confusionMatrix(as.factor(test$y), as.factor(test$RF_Pred))

#########################################################################
#############  Model after removing unknown observations ################
#########################################################################

#Missing values
lapply(bank,unique)
#unknowns in job, marital, education, default, housing, loan, 
miss = subset(bank, bank$job == "unknown")#39 observations 
miss = subset(bank, bank$marital == "unknown")#11 observations 
miss = subset(bank, bank$education == "unknown")#167 observations 
miss = subset(bank, bank$default == "unknown")#803 observations 
miss = subset(bank, bank$housing == "unknown")#105 observations 
miss = subset(bank, bank$loan == "unknown")#105 observations 

#Clean 
#Remove duration because when duration=0, non-contacted and therefore, perfectly correlated with y
bank = bank[,!(names(bank) %in% "duration")]
#Remove single illiterate education observation because if model not trained with, gives error when predicting on test data
bank = subset(bank, !(bank$education == "illiterate")) #removed 1 observation
#pdays=999 means not previously contacted
#Create new factor variable for pdays ("recently contacted, "not contacted" etc.) 
bank$pdaysdummy = ifelse(bank$pdays == 999, "Not contacted", 
                         ifelse(bank$pdays <= 7, "1 Week",
                                ifelse(bank$pdays <= 14, "2 Weeks", 
                                       ifelse(bank$pdays <= 21, "3 Weeks", "3+ Weeks"))))
bank$pdaysdummy  = as.factor(bank$pdaysdummy)
#Remove unknown observations
bank = subset(bank, !(bank$job == "unknown")) 
bank = subset(bank, !(bank$marital == "unknown")) 
bank = subset(bank, !(bank$education == "unknown")) 
bank = subset(bank, !(bank$default == "unknown")) 
bank = subset(bank, !(bank$housing == "unknown")) 
bank = subset(bank, !(bank$loan == "unknown")) 



#Train/Test Split
set.seed(1)
train_partition = createDataPartition(bank$y, p = 0.8)[[1]]
train  = bank[train_partition,]
test   = bank[-train_partition,]

#Logistic model with all variables
a1 = glm(y ~ . -pdays, data = train, family = binomial)
summary(a1)

#Predictions
test$PredProb = predict.glm(a1, newdata = test, type = "response")
#Convert probabilities to 1s and 0s
test$PredSur = ifelse(test$PredProb >= 0.85, "yes", "no") #Adjusted prob to increase specificity 

# Finally, we will use the command "confusionMatrix" from the package caret to get accuracy of the model prediction. 
caret::confusionMatrix(as.factor(test$y), as.factor(test$PredSur)) #Comparing observed to predicted

### Random Forest Model ###
set.seed(1)
rf_model = train(y ~ . -pdays, data = train, method = "rf", trControl = trainControl(method = "cv", number = 10))
print(rf_model)

# Predictions for random forest
test$RF_Pred = predict(rf_model, newdata = test)

# Confusion matrix for random forest
caret::confusionMatrix(as.factor(test$y), as.factor(test$RF_Pred))

