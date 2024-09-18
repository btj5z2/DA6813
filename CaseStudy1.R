pacman::p_load(caret, lattice, tidyverse, gam, logistf, MASS, car, corrplot, gridExtra, ROCR, RCurl, randomForest)

##For lit review, write a paper that contains an analysis on bank-related data and compare what analytical techniques they used and worked

##### Data Set ######

#bank = read.csv("E:/UTSA/DA6813 Data Analytics Applications/Case Study 1/bank-additional.csv", sep = ";")
bank = as.data.frame(read.csv(text = getURL('https://raw.githubusercontent.com/btj5z2/DA6813/main/bank-additional.csv'), sep = ';'))
str(bank)

#Copy of data set to model
b = bank

#Turning character variables into factors
fac_vars = c("job", "marital", "education", "default", "housing", "loan", "contact", 
             "month", "day_of_week", "poutcome", "y")
b[fac_vars] = lapply(b[fac_vars],as.factor)


##### Balanced? No.##### 
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

box_emp.var.rate = ggplot(bank, aes(y, emp.var.rate)) +
  geom_boxplot()

box_cons.price.idx = ggplot(bank, aes(y, cons.price.idx)) +
  geom_boxplot()

box_cons.conf.idx = ggplot(bank, aes(y, cons.conf.idx)) +
  geom_boxplot()

box_euribor3m = ggplot(bank, aes(y, euribor3m)) +
  geom_boxplot()

box_nr.employed = ggplot(bank, aes(y, nr.employed)) +
  geom_boxplot()

grid.arrange(box_age, box_duration, box_campaign, box_pdays, box_previous,
             box_emp.var.rate, box_cons.price.idx, box_cons.conf.idx, box_euribor3m,
             box_nr.employed,
             ncol = 5)

# side-by-side bar plots for categorical variables

p = ggplot(bank) +
  facet_wrap(~y) +
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
vif(lm(bank[,c(1,11:14,16:20)])) #3 numeric columns with high VIF (i.e. >10) : emp.var.rate, euribor3m, and nr.employed (euribor3m w/ the highest)
vif(lm(bank[,c(1,11:14,16:18,20)])) #Removed euribor3m
vif(lm(bank[,c(1,11:14,17:18,20)])) #Removed emp.var.rate

b = b[,!(names(b) %in% "euribor3m")]
b = b[,!(names(b) %in% "emp.var.rate")]


# Identify Missing values
lapply(bank,unique)
#unknowns in job, marital, education, default, housing, loan, 
miss = subset(bank, bank$job == "unknown")#39 observations 
miss = subset(bank, bank$marital == "unknown")#11 observations 
miss = subset(bank, bank$education == "unknown")#167 observations 
miss = subset(bank, bank$default == "unknown")#803 observations 
miss = subset(bank, bank$housing == "unknown")#105 observations 
miss = subset(bank, bank$loan == "unknown")#105 observations 

##### Clean ##### 
#Remove duration because when duration=0, non-contacted and therefore, perfectly correlated with y
b = b[,!(names(b) %in% "duration")]
#Remove single illiterate education observation because if model not trained with, gives error when predicting on test data
b = subset(b, !(b$education == "illiterate")) #removed 1 observation
#pdays=999 means not previously contacted
#Create new factor variable for pdays ("recently contacted, "not contacted" etc.) 
b$pdaysdummy = ifelse(b$pdays == 999, "Not contacted", 
                         ifelse(b$pdays <= 7, "1 Week",
                                ifelse(b$pdays <= 14, "2 Weeks", 
                                       ifelse(b$pdays <= 21, "3 Weeks", "3+ Weeks"))))
b$pdaysdummy  = as.factor(b$pdaysdummy)
ggplot(b) +
  facet_wrap(~y) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  geom_bar(aes(x = b$pdaysdummy))


#Train/Test Split
set.seed(1)
train_partition = createDataPartition(b$y, p = 0.8)[[1]]
train  = b[train_partition,]
test   = b[-train_partition,]

#Logistic model with all variables
a1 = step(glm(y ~ . -pdays, data = train, family = binomial), direction = "backward")
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

#Copy of data set to remove unknown observations
b1 = b

#Remove unknown observations
b1 = subset(b1, !(b1$job == "unknown")) 
b1 = subset(b1, !(b1$marital == "unknown")) 
b1 = subset(b1, !(b1$education == "unknown")) 
b1 = subset(b1, !(b1$default == "unknown")) 
b1 = subset(b1, !(b1$housing == "unknown")) 
b1 = subset(b1, !(b1$loan == "unknown")) 

#Train/Test Split
set.seed(1)
train_partition1 = createDataPartition(b1$y, p = 0.8)[[1]]
train1  = b1[train_partition1,]
test1   = b1[-train_partition1,]

#Logistic model with all variables
a2 = step(glm(as.factor(y) ~ . -pdays, data = train1, family = binomial), direction = "backward")
summary(a2)

#Predictions
test1$PredProb = predict.glm(a2, newdata = test1, type = "response")
#Convert probabilities to 1s and 0s
test1$PredSur = ifelse(test1$PredProb >= 0.85, "yes", "no") #Adjusted prob to increase specificity 

# Finally, we will use the command "confusionMatrix" from the package caret to get accuracy of the model prediction. 
caret::confusionMatrix(as.factor(test1$y), as.factor(test1$PredSur)) #Comparing observed to predicted

### Random Forest Model ###
set.seed(1)
rf_model1 = train(as.factor(y) ~ . -pdays, data = train1, method = "rf", trControl = trainControl(method = "cv", number = 10))
print(rf_model1)

# Predictions for random forest
test1$RF_Pred = predict(rf_model1, newdata = test1)

# Confusion matrix for random forest
caret::confusionMatrix(as.factor(test1$y), as.factor(test1$RF_Pred))
