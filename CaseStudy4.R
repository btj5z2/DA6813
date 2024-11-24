# Libraries
pacman::p_load(caret, lattice, tidyverse, gam, logistf, MASS, 
               car, corrplot, gridExtra, ROCR, RCurl, randomForest, 
               readr, readxl, e1071, klaR, bestNormalize, rpart, lubridate,
               tseries, quantmod, knitr, SMCRM, tree, rpart.plot)


# Data
data(acquisitionRetention)
crm = acquisitionRetention


## EDA
str(crm)

### customer    - customer number (from 1 to 500)
### acquisition - 1 if the prospect was acquired, 0 otherwise - RESPONSE
### duration    - number of days the customer was a customer of the firm, 
###               0 if acquisition == 0 - RESPONSE
### profit      - customer lifetime value (CLV) of a given customer, 
###               -(Acq_Exp) if the customer is not acquired
### acq_exp     - total dollars spent on trying to acquire this prospect
### ret_exp     - total dollars spent on trying to retain this customer
### acq_exp_sq  - square of the total dollars spent on trying to acquire this prospect
### ret_exp_sq  - square of the total dollars spent on trying to retain this customer
### freq        - number of purchases the customer made during that customer's 
###               lifetime with the firm, 0 if acquisition == 0
### freq_sq     - square of the number of purchases the customer made during 
###               that customer's lifetime with the firm
### crossbuy    - number of product categories the customer purchased from during 
###               that customer's lifetime with the firm, 0 if acquisition = 0
### sow         - Share-of-Wallet; percentage of purchases the customer makes 
###               from the given firm given the total amount of purchases across all firms in that category
### industry    - 1 if the customer is in the B2B industry, 0 otherwise
### revenue     - annual sales revenue of the prospect's firm (in millions of dollar)
### employees   - number of employees in the prospect's firm


### Convert data type for factor variables
fac_vars = c('acquisition', 'industry')
crm[fac_vars] = lapply(crm[fac_vars],as.factor)


### Drop customer field since it is an ID number unique to the customer
crm = crm %>%
  dplyr::select(-customer)


### Check for NA
which(is.na(crm)) # No NA's found


### Viz features - Acquisition
grid.arrange(
  ggplot(crm, aes(acquisition, duration)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, profit)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, acq_exp)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, ret_exp)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, acq_exp_sq)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, ret_exp_sq)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, freq)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, freq_sq)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, crossbuy)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, sow)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, revenue)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, employees)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, after_stat(count))) + geom_bar(aes(fill = industry), position = 'dodge'),
  bottom = 'Figure X.X: Plots of predictor relationship with acquisition response'
)
# duration, ret_exp, ret_exp_sq, freq, freq_sq, crossbuy, and sow are perfect predictors
# all of these features will only return a value if the customer is acquired
# otherwise, these are 0
# Also removed profit b/c it is negative number if not acquired 


### Create acquisition data set with perfect predictors removed
crm_acq = crm %>%
              dplyr::select(-c(duration, profit, ret_exp, ret_exp_sq, freq, freq_sq, crossbuy, sow))
str(crm_acq)


### Viz Features without relationship to a response
grid.arrange(
  ggplot(crm, aes(profit)) + geom_histogram(bins = 30),
  ggplot(crm, aes(acq_exp)) + geom_histogram(bins = 30),
  ggplot(crm, aes(ret_exp)) + geom_histogram(bins = 30),
  ggplot(crm, aes(acq_exp_sq)) + geom_histogram(bins = 30),
  ggplot(crm, aes(ret_exp_sq)) + geom_histogram(bins = 30),
  ggplot(crm, aes(freq)) + geom_histogram(bins = 30),
  ggplot(crm, aes(freq_sq)) + geom_histogram(bins = 30),
  ggplot(crm, aes(crossbuy)) + geom_histogram(bins = 30),
  ggplot(crm, aes(sow)) + geom_histogram(bins = 30),
  ggplot(crm, aes(industry)) + geom_bar(),
  ggplot(crm, aes(revenue)) + geom_histogram(bins = 30),
  ggplot(crm, aes(employees)) + geom_histogram(bins = 30),
  bottom = 'Figure X.X: Plots of predictor variables'
)


### Viz response variables - Acquisition
crm_acq %>%
  ggplot(aes(acquisition)) +
  geom_bar()
# data is imbalanced


### Check for multicollinearity - Acquisition 
lin.model = glm(acquisition~ . , data = crm_acq, family=binomial())
vif(lin.model) #Remove acq_exp_sq
lin.model = glm(acquisition~ . -acq_exp_sq , data = crm_acq, family=binomial())
vif(lin.model) #All VIF<5
#Remove acq_exp from data set
crm_acq = crm %>%
  dplyr::select(-c(duration, profit, ret_exp, ret_exp_sq, freq, freq_sq, crossbuy, sow, acq_exp_sq))


##### Corr Plot - Acquisition
num_cols = crm_acq[,sapply(crm_acq, is.numeric)]
corrplot::corrplot(cor(num_cols), method = c("number"))


### Balance and create training/testing data sets for Acquisition models 
#Below is balancing the data set by taking all 0 observations and randomly sample 162 of the 338 acquired customers. 
#This data set (324 obs) can be split into a training and testing data sets (80%/20%) for the customer acquisition models    

set.seed(123)
train_1 = crm_acq %>% filter(acquisition ==1) #338 observations of acquired customers
train_0 = crm_acq %>% filter(acquisition ==0) #162 observations of non-acquired customers

sample_1 = sample_n(train_1, nrow(train_0)) #randomly sampling "train_1" the number of observations in "train_0"
crm_acq_bal = rbind(train_0, sample_1) #complete data set for the acquisition models 
acq_partition = createDataPartition(crm_acq_bal$acquisition, p = 0.8)[[1]] #create 80% split 
train_acq  = crm_acq_bal[acq_partition,] #training data set to be used on acquisition models
test_acq   = crm_acq_bal[-acq_partition,] #testing data set to be used on acquisition models


######### Acquisition Model #########   
## Logistic regression
log.model = glm(acquisition ~ . , data = train_acq, family = binomial) 
summary(log.model)
stepwise_log_model = step(log.model, direction = "both")
summary(stepwise_log_model) #acq_exp removed (not being significant)
test_acq$PredPercent = predict.glm(stepwise_log_model, newdata = test_acq, type = "response") #predictions
test_acq$PredPercent_binary = ifelse(test_acq$PredPercent>0.50, 1, 0)
actual_vs_pred = data.frame(Actual = test_acq$acquisition, Predicted = test_acq$PredPercent_binary)

# Confusion Matrix to compare actual vs predicted directional movements
test_acq$PredPercent_binary = as.factor(test_acq$PredPercent_binary)
conf_matrix_glm = caret::confusionMatrix(test_acq$PredPercent_binary, test_acq$acquisition, positive = "1")
print(conf_matrix_glm)

# accuracy, sensitivity, and specificity
accuracy = conf_matrix_glm$overall["Accuracy"]
sensitivity = conf_matrix_glm$byClass["Sensitivity"]
specificity = conf_matrix_glm$byClass["Specificity"]

print(paste("Accuracy:", accuracy))
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))


## Decision Trees for Acquired Data
# Set seed for reproducibility
set.seed(123)
tree.acq2 = tree(acquisition ~ ., data = train_acq)
summary(tree.acq2)

# Plot the tree 
plot(tree.acq2)
text(tree.acq2, cex=0.7)

# Build a regression tree model using rpart with method = "class"
decision_tree_model = rpart(acquisition ~ . -acq_exp, data = train_acq, method = "class")
summary(decision_tree_model)
#Check variable importance scores
importance = decision_tree_model$variable.importance #implies acq_exp is least important variable. 
#Removed acq_exp so the same variables are used as in logisitic regression model
print(importance)
# visualization
plot(decision_tree_model)                
text(decision_tree_model, cex=0.7)   

rpart.plot(decision_tree_model, 
           type = 3,                     # Display both node labels and probabilities
           extra = 104,                  # Add details like class probabilities and percentages
           under = TRUE,                 # Show percentages below the node
           fallen.leaves = TRUE,         # Align leaf nodes
           main = "Decision Tree for Acquisition")



# Predict on the test set with continuous values
dt_predictions = predict(decision_tree_model, test_acq, type= "class")

conf_matrix = table(dt_predictions, test_acq$acquisition)

true_positive = conf_matrix[2, 2]  # Correctly predicted "Yes" (or 1)
true_negative = conf_matrix[1, 1]  # Correctly predicted "No" (or 0)
false_positive = conf_matrix[2, 1]  # Incorrectly predicted "Yes" (or 1)
false_negative = conf_matrix[1, 2]  # Incorrectly predicted "No" (or 0)

# Calculate metrics
accuracy_tree = sum(diag(conf_matrix)) / sum(conf_matrix)
sensitivity_tree = true_positive / (true_positive + false_negative)  # True Positive Rate
specificity_tree = true_negative / (true_negative + false_positive)  # True Negative Rate

# Print results
print(paste("Accuracy:", round(accuracy_tree, 4)))
print(paste("Sensitivity:", round(sensitivity_tree, 4)))
print(paste("Specificity:", round(specificity_tree, 4)))


### PRUNING 
plotcp(decision_tree_model)

optimal_cp = decision_tree_model$cptable[which.min(decision_tree_model$cptable[, "xerror"]), "CP"]
pruned_tree = prune(decision_tree_model, cp = optimal_cp)
#Check variable importance scores
importance = pruned_tree$variable.importance #implies acq_exp is least important variable
print(importance)
plot(pruned_tree)                
text(pruned_tree, cex=0.7) 

rpart.plot(pruned_tree, 
           type = 3,                     # Display both node labels and probabilities
           extra = 104,                  # Add details like class probabilities and percentages
           under = TRUE,                 # Show percentages below the node
           fallen.leaves = TRUE,         # Align leaf nodes
           main = "Pruned Decision Tree for Acquisition")


# Predict on test data
pruned_predictions = predict(pruned_tree, test_acq, type = "class")

# Generate confusion matrix and compute metrics
pruned_conf_matrix = table(pruned_predictions, test_acq$acquisition)
true_positive_prune = conf_matrix[2, 2]  # Correctly predicted "Yes" (or 1)
true_negative_prune = conf_matrix[1, 1]  # Correctly predicted "No" (or 0)
false_positive_prune = conf_matrix[2, 1]  # Incorrectly predicted "Yes" (or 1)
false_negative_prune = conf_matrix[1, 2]  # Incorrectly predicted "No" (or 0)

# Calculate metrics
accuracy_tree2 = sum(diag(conf_matrix)) / sum(conf_matrix)
sensitivity_tree2 = true_positive / (true_positive + false_negative)  # True Positive Rate
specificity_tree2 = true_negative / (true_negative + false_positive)  # True Negative Rate

# Print results
print(paste("Accuracy:", accuracy_tree2))
print(paste("Sensitivity:", sensitivity_tree2))
print(paste("Specificity:", specificity_tree2))


######### Customer acquisition predictions on whole data set #########
#Logistic regression had highest performance metrics so we are moving forward with it 
crm$PredPercent = predict.glm(stepwise_log_model, newdata = crm, type = "response") #predictions
crm$PredPercent_binary = ifelse(crm$PredPercent>0.50, 1, 0)


######### Duration Model #########  
### Duration sub-group
crm_dur = crm %>%
  filter(PredPercent_binary == 1) %>% # filter out un-acquired customers
  dplyr::select(-c(acquisition, PredPercent, PredPercent_binary)) # drop acquisition and prediction variables

### Viz features - Duration
grid.arrange(
  ggplot(crm_dur, aes(x = duration, y = profit)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = acq_exp)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = ret_exp)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = acq_exp_sq)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = ret_exp_sq)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = freq)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = freq_sq)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = crossbuy)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = sow)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = revenue)) + geom_point(),
  ggplot(crm_dur, aes(x = duration, y = employees)) + geom_point(),
  ggplot(crm_dur, aes(industry, duration)) + geom_boxplot(),
  bottom = 'Figure X.X: Plots of predictor relationship with duration response'
)

### Viz response variables - Duration
crm_dur %>%
  ggplot(aes(duration)) +
  geom_histogram(bins = 30)


### Check for multicollinearity - Duration
lin.model = glm(duration~ . , data = crm_dur)
vif(lin.model) #Remove ret_exp
lin.model = glm(duration~ . -ret_exp, data = crm_dur)
vif(lin.model) #Remove acq_exp
lin.model = glm(duration~ . -ret_exp -acq_exp, data = crm_dur)
vif(lin.model) #Remove freq
lin.model = glm(duration~ . -ret_exp -acq_exp -freq, data = crm_dur)
vif(lin.model) #Remove profit (VIF of 5.5, figured we still have several other variables so was removed)
lin.model = glm(duration~ . -ret_exp -acq_exp -freq -profit, data = crm_dur)
vif(lin.model) #All VIFs<5
#Remove variables accordingly
crm_dur = crm_dur %>%
  dplyr::select(-c(ret_exp, acq_exp, freq, profit))


##### Corr Plot - Duration
num_cols = crm_dur[,sapply(crm_dur, is.numeric)]
corrplot::corrplot(cor(num_cols), method = c("number"))


#For the duration models, we can use all acquired customers as then create train & test split (80/20)
dur_partition = createDataPartition(crm_dur$duration, p = 0.8)[[1]]
train_dur  = crm_dur[dur_partition,] #training data set to be used on duration models
test_dur   = crm_dur[-dur_partition,] #testing data set to be used on duration models


# Train the Random Forest model
rf_model <- randomForest(duration ~ ., data = train_dur, importance = TRUE, ntree = 100)
print(rf_model)

# Make predictions on the test set
predictions <- predict(rf_model, newdata = test_dur)

# Check performance using metrics like RMSE, R-squared, and MAE
rmse <- sqrt(mean((predictions - test_dur$duration)^2))
ss_total <- sum((test_dur$duration - mean(test_dur$duration))^2)
ss_residual <- sum((test_dur$duration - predictions)^2)
r_squared <- 1 - (ss_residual / ss_total)
mae <- mean(abs(predictions - test_dur$duration))
print(paste("RMSE: ", round(rmse,2)))
print(paste("R-squared: ", round(r_squared,2)))
print(paste("MAE: ", round(mae,2)))

# Visualize variable importance
importance_rf <- importance(rf_model)
varImpPlot(rf_model)  
#Tried re-running random forest without the least important variables (revenue, industry, & employees)
#but got slightly worse number (i.e. 94% var explained, 101 RMSE, 0.93 R2, and 63 MAE)

#Tuned Random Forest
tune_grid <- expand.grid(.mtry = 6:10)  # Set range for mtry
rf_tune <- train(duration ~ ., data = train_dur, method = "rf", tuneGrid = tune_grid)
print(rf_tune)
#The tuning process showed that the optimal mtry value is 9, which suggests that using 9 variables 
#at each split improves model performance. 

predictions <- predict(rf_tune, newdata = test_dur)

# Check performance using metrics like RMSE, R-squared, and MAE
rmse <- sqrt(mean((predictions - test_dur$duration)^2))
ss_total <- sum((test_dur$duration - mean(test_dur$duration))^2)
ss_residual <- sum((test_dur$duration - predictions)^2)
r_squared <- 1 - (ss_residual / ss_total)
mae <- mean(abs(predictions - test_dur$duration))
print(paste("RMSE: ", round(rmse,2)))
print(paste("R-squared: ", round(r_squared,4)))
print(paste("MAE: ", round(mae,2)))


####### PDP PLOTS#######
library("pdp")
pdp_ret_exp_sq <- partial(rf_tune, pred.var = "ret_exp_sq", plot = TRUE,
                       train = train_dur, main = "PDP for ret_exp_sq")

#plot(pdp_ret_exp_sq)

pdp_crossbuy <- partial(rf_tune, pred.var = "crossbuy", plot = TRUE,
                          train = train_dur, main = "PDP for crossbuy")

#plot(pdp_crossbuy)

pdp_industry <- partial(rf_tune, pred.var = "industry", plot = TRUE,
                        train = train_dur, main = "PDP for industry")

#plot(pdp_industry)

pdp_acq_exp_sq <- partial(rf_tune, pred.var = "acq_exp_sq", plot = TRUE,
                        train = train_dur, main = "PDP for acq_exp_sq")

#plot(pdp_acq_exp_sq)

pdp_sow <- partial(rf_tune, pred.var = "sow", plot = TRUE,
                          train = train_dur, main = "PDP for sow")

#plot(pdp_sow)

pdp_freq_sq <- partial(rf_tune, pred.var = "freq_sq", plot = TRUE,
                   train = train_dur, main = "PDP for freq_sq")

#plot(pdp_freq_sq)

pdp_revenue <- partial(rf_tune, pred.var = "revenue", plot = TRUE,
                   train = train_dur, main = "PDP for revenue")

#plot(pdp_revenue)

pdp_emp <- partial(rf_tune, pred.var = "employees", plot = TRUE,
                       train = train_dur, main = "PDP for employees")

#plot(pdp_emp)

grid.arrange(pdp_emp, pdp_revenue, pdp_freq_sq, pdp_sow, pdp_acq_exp_sq, pdp_industry, pdp_crossbuy, pdp_ret_exp_sq, ncol = 2)

