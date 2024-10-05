pacman::p_load(caret, lattice, tidyverse, gam, logistf, MASS, car, corrplot, gridExtra, ROCR, RCurl, randomForest, readr, readxl, e1071, klaR)

##For lit review, write a paper that contains an analysis on bank-related data and compare what analytical techniques they used and worked
#Sample training data set like done in titanic example

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

#Merge both datasets for visualizations
bbbc_full = union_all(test, train)

##### Balanced? No.##### 
ggplot(bbbc_full, aes(Choice)) +
  geom_bar() +
  labs(caption = "Figure X.X:") +
  theme(plot.caption = element_text(hjust = 0.5))
  
#Visualizations
##Numerical Variables
grid.arrange(ggplot(bbbc_full, aes(Choice, Amount_purchased)) + geom_boxplot(),
             ggplot(bbbc_full, aes(Choice, Frequency)) + geom_boxplot(),
             ggplot(bbbc_full, aes(Choice, Last_purchase)) + geom_boxplot(),
             ggplot(bbbc_full, aes(Choice, First_purchase)) + geom_boxplot(),
             ggplot(bbbc_full, aes(Choice, P_Child)) + geom_boxplot(),
             ggplot(bbbc_full, aes(Choice, P_Youth)) + geom_boxplot(),
             ggplot(bbbc_full, aes(Choice, P_Cook)) + geom_boxplot(),
             ggplot(bbbc_full, aes(Choice, P_DIY)) + geom_boxplot(),
             ggplot(bbbc_full, aes(Choice, P_Art)) + geom_boxplot(),
             ncol = 5,
             bottom = 'Figure X.X: Boxplots of Numerical Values')

##Categorical Variables
ggplot(bbbc_full) + 
  facet_wrap(~Choice) +
  labs(caption = "Figure X.X:") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1), plot.caption = element_text(hjust = 0.5)) + 
  geom_bar(aes(x = Gender))

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
str(test) #"test" data set should be used on predictions

### BALANCE DATA
set.seed(123)
train_y = train %>% filter(Choice ==1)
train_n = train %>% filter(Choice ==0)

sample_y = sample_n(train_n, nrow(train_y))
train_bal = rbind(train_y, sample_y) #"train_bal" is final trained data set to be used on models 

plot(train_bal$Choice)

#Balancing test test data set not needed
#test_y = test %>% filter(Choice ==1)
#test_n = test %>% filter(Choice ==0)

#sample_test_y = sample_n(test_n, nrow(test_y))
#test_bal = rbind(test_y, sample_test_y)

#plot(test_bal$Choice)

### Linear Model
## LR Model
lr_train = train_bal
lr_train$Choice <- as.numeric(as.character(lr_train$Choice))  # it's a factor stored as numbers
lr_train$Gender = as.numeric(as.character(lr_train$Gender))
lr_test = test
lr_test$Choice = as.numeric(as.character(lr_test$Choice))
lr_test$Gender = as.numeric(as.character(lr_test$Gender))
m1 = lm(Choice ~., data = lr_train)
vif(m1) #last_purchase high vif
m2 = lm(Choice ~ . -Last_purchase, data = lr_train)
vif(m2) #first_purchase vif > 5
m3 <- lm(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Youth + P_Cook + P_DIY + P_Art, 
         data = lr_train)
vif(m3) #vifs look good 

summary(m3) #P_Youth not significant 

m4 = lm(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Cook + P_DIY + P_Art, 
        data = lr_train)
summary(m4)

#P_Cook became significant after training the model on balanced training data set 
#m5 = lm(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_DIY + P_Art, 
        #data = train)
#summary(m5) 


predictions = predict(m4, newdata = lr_test, type = "response")



#Measures
mse = mean((lr_test$Choice - predictions)^2)
mae = mean(abs(lr_test$Choice - predictions))
me = mean(lr_test$Choice - predictions)
mape =  mean(abs(lr_test$Choice - predictions)/lr_test$Choice)*100






#Multi Collinearity
log.model = glm(Choice ~ . , data = train_bal, family = binomial)
vif(log.model)

log.model2 = glm(Choice ~ . -Last_purchase , data = train_bal, family = binomial) #Remove last_purchased
vif(log.model2)

log.model3 = glm(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Youth + P_Cook + P_DIY 
                 + P_Art , data = train_bal, family = binomial) #Remove first_purchased
vif(log.model3) #all vifs<5

#Logistic model 
summary(log.model3) #P_Youth not significant

log.model4 = glm(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Cook + P_DIY 
                 + P_Art , data = train_bal, family = binomial) #Remove P_Youth
summary(log.model4)

#Predictions 
test$PredProb = predict.glm(log.model4, newdata = test, type = "response")
test$PredSur = ifelse(test$PredProb >= 0.54, 1, 0) # Create new variable converting probabilities to 1s and 0s

# "Confusion Matrix" to get accuracy of the model prediction
caret::confusionMatrix(as.factor(test$PredSur), as.factor(test$Choice),positive = "1" ) #Comparing observed to predicted


### Adding SVM Model (with balanced data and optimal predictors) ###



# We'll scale both the training and testing data (except for the target variable)
train_scaled <- train_bal
test_scaled <- test

# Scale the numeric columns in the training and test sets
test_scaled$Choice <- as.factor(test_scaled$Choice)
train_scaled$Choice <- as.factor(train_scaled$Choice)
numeric_cols <- sapply(train_scaled, is.numeric)
train_scaled[numeric_cols] <- scale(train_scaled[numeric_cols])
test_scaled[numeric_cols] <- scale(test_scaled[numeric_cols])


# SVM Model with Grid Search for Hyper parameter Tuning
# Define tuning grid for cost and gamma (for radial kernel)
tune_grid <- expand.grid(C = c(0.01, 0.1, 1, 10, 100), 
                         sigma = c(0.001, 0.01, 0.1, 1))

# Tuning Radial Basis Function (RBF) kernel SVM
set.seed(123)
svm_tune <- tune(svm, Choice ~ ., data = train_scaled, 
                 kernel = "radial", 
                 ranges = list(cost = tune_grid$C, gamma = tune_grid$sigma),
                 scale = FALSE)  # Don't scale inside the SVM function, we've already done that

# Best model based on cross-validation
best_model <- svm_tune$best.model
print(svm_tune)

# Test the tuned SVM model
pred_svm <- predict(best_model, newdata = test_scaled)
confusionMatrix(pred_svm, test_scaled$Choice)

# Testing Other Kernels
# Linear Kernel
set.seed(123)
svm_linear <- svm(Choice ~ ., data = train_scaled, 
                  kernel = "linear", 
                  cost = best_model$cost, 
                  scale = FALSE)
pred_linear <- predict(svm_linear, newdata = test_scaled)
confusionMatrix(pred_linear, test_scaled$Choice)

# Polynomial Kernel (degree 3)
set.seed(123)
svm_poly <- svm(Choice ~ ., data = train_scaled, 
                kernel = "polynomial", 
                cost = best_model$cost, 
                degree = 3, 
                scale = FALSE)
pred_poly <- predict(svm_poly, newdata = test_scaled)
confusionMatrix(pred_poly, test_scaled$Choice)

# Compare accuracy across different kernels
linear_acc <- sum(pred_linear == test_scaled$Choice) / nrow(test_scaled)
rbf_acc <- sum(pred_svm == test_scaled$Choice) / nrow(test_scaled)
poly_acc <- sum(pred_poly == test_scaled$Choice) / nrow(test_scaled)

cat("Linear Kernel Accuracy: ", linear_acc, "\n")
cat("RBF Kernel Accuracy: ", rbf_acc, "\n")
cat("Polynomial Kernel Accuracy: ", poly_acc, "\n")


### Adding SVM Model ###

# SVM requires normalized numerical variables
# Refit SVM model using e1071 package
svm.model <- svm(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Cook + P_DIY + P_Art, 
                        data = train, 
                        kernel = "radial",  
                        probability = TRUE)

# Predict using the test data
test$svm_pred <- predict(svm.model, test, probability = TRUE)
svm.prob <- attr(predict(svm.model, test, probability = TRUE), "probabilities")[,2]

# Classify predictions based on a threshold of 0.5
test$svm_class <- ifelse(svm.prob >= 0.5, 1, 0)

# Evaluate SVM predictions
caret::confusionMatrix(as.factor(test$Choice), as.factor(test$svm_class))


### LDA Model ###

# LDA Feature Selection
rand_f.model = randomForest::randomForest(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Youth + P_Cook + P_DIY + P_Art, data = train_bal)

varImpPlot(rand_f.model,
           sort = T,
           n.var = 10,
           main = "Top 10 - Variable Importance")

# LDA Model
LDA.model.full = lda(Choice ~ Gender + Amount_purchased + Frequency + P_Child + P_Youth + P_Cook + P_DIY + P_Art, data = train_bal)

LDA.model.trim = lda(Choice ~ Amount_purchased + Frequency + P_Art, data = train_bal)

# LDA Predictions
LDA.preds.full = predict(LDA.model.full, test, probability = TRUE)

LDA.preds.trim = predict(LDA.model.trim, test, probability = TRUE)

# Model Performance
## All Predictors
caret::confusionMatrix(as.factor(LDA.preds.full$class), test$Choice, positive = '1')

## Trimmed Predictors
caret::confusionMatrix(as.factor(LDA.preds.trim$class), test$Choice, positive = '1')


##### Predicting Profits ##### 
#Calc profit per book 
book_cost = 15*1.45 #each book costs $15 plus 45% overhead
book_profit = 31.95-book_cost #each book is sold for $31.95

#If sending mailings to entire list
total_list = 50000 #list of 50,000 customers to mail ad to 
total_cost = 0.65*total_list #cost to mail entire list an ad 
#According to case study prompt, 9.03% of mailings resulted in an order
total_profit = 0.0903*total_list*book_profit-total_cost

#If using logistic regression model to selectively send mailings
#predicted (599+146)/(2300)=32.4% would purchase a book with only 146/(599+146) actually buying the book (19.6%)
model_list = total_list*0.324 #Number of customers to send mailings to
model_cost = 0.65*model_list #$0.65/mailing
model_profit = 0.196*model_list*book_profit-model_cost

#If using LDA model to selectively send mailings
#predicted (751 + 150) / 2300 = 39.17% would purchase a book with only 150 / (751 + 150) of those customers actually buying the book (16.6%)
model_list_lda = total_list * ((751 + 150) / 2300)
model_cost_lda = 0.65 * model_list_lda
model_profit_lda = (150 / (751 + 150)) * model_list_lda * book_profit - model_cost_lda

#Summary of results
cat("Profit if mailed entire list: $", total_profit, "\n")
cat("Profit if list generated by Logistic model: $", model_profit, "\n")
cat("% Increase in Profits: ", (model_profit-total_profit)*100/total_profit, "%", "\n")

cat("Profit if list generated by LSA model: $", model_profit_lda, '\n')
cat('% increase in Profits with LDA model: ', (model_profit_lda - total_profit) * 100 / total_profit, '%', '\n')
