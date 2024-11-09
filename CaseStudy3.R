pacman::p_load(caret, lattice, tidyverse, gam, logistf, MASS, 
               car, corrplot, gridExtra, ROCR, RCurl, randomForest, 
               readr, readxl, e1071, klaR, bestNormalize, rpart, lubridate,
               tseries, quantmod)

##### Data Set ######
dow_raw = as.data.frame(read.csv(text = getURL('https://raw.githubusercontent.com/btj5z2/DA6813/main/dow_jones_index.data'), header = TRUE))
sp500_raw = as.data.frame(read.csv(text = getURL('https://raw.githubusercontent.com/btj5z2/DA6813/refs/heads/main/SP500.csv'), header = TRUE))

##### Copy of Data Set for Model ######
dow = dow_raw
sp500 = sp500_raw

### Review Details of Data Set ###
str(dow)
str(sp500)

# Many numeric values were read in as strings
# Convert these values to numeric data types

num_vars = c('open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close')
dow[num_vars] = lapply(dow[num_vars], gsub, pattern = '[\\$,]', replacement = '')
dow[num_vars] = lapply(dow[num_vars], as.numeric)

sp500 =
  sp500 %>% rename(
    date      = Date,
    open      = Open,
    high      = High,
    low       = Low,
    close     = Close.,
    adj_close = Adj.Close.,
    volume = Volume
  )
  
sp500_num_vars = c('open', 'high', 'low', 'close', 'adj_close', 'volume')
sp500[sp500_num_vars] = lapply(sp500[sp500_num_vars], gsub, pattern = '[\\$,]', replacement = '')
sp500[sp500_num_vars] = lapply(sp500[sp500_num_vars], as.numeric)

# Convert 'date' column to date type

dow$date = as.Date(dow$date, '%m/%d/%Y')
sp500$date = dmy(sp500$date)

sp500 = sp500 %>%
  arrange(date)

# Get SP500 weekly returns

sp500$percent_change_next_weeks_price = Delt(sp500[,5], type = 'arithmetic') * 100

# Convert 'stock' and 'quarter' column to factor type

#dow$stock = as.factor(dow$stock)
#dow$quarter = as.factor(dow$quarter)

### Review column details to validate changes ###
str(dow)
str(sp500)

### Identify columns with NA's ###
names(which(colSums(is.na(dow)) > 0))

paste('Total NAs in percent_change_volume_over_last_wk: ', sum(is.na(dow$percent_change_volume_over_last_wk)))
paste('Total NAs in previous_weeks_volume: ', sum(is.na(dow$previous_weeks_volume)))

## Number of columns where percent_change_volume_over_last_wk is NA while previous_weeks_volume is not
dow %>%
  tally(is.na(dow$percent_change_volume_over_last_wk) & !is.na(dow$previous_weeks_volume))

## Number of columns where both percent_change_volume_over_last_wk and previous_weeks_volume are NA
dow %>%
  tally(is.na(dow$percent_change_volume_over_last_wk) & is.na(dow$previous_weeks_volume))

# percent_change_volume_over_last_wk and previous_weeks_volume are both either NA or not NA
# this makes sense because the percent change would have to be null if there was no previous week volume

# Show which dates have NA values for previous week
unique(dow$date[is.na(dow$percent_change_volume_over_last_wk)])

# Count number of entries of the date that includes NA values
table(dow$date[dow$date=="2011-01-07"])

# Check for earliest date in dataset
min(dow$date)

# All entries on 1/7/2011 have NA values for prior week.
# This date is the earliest date in the dataset, so these values being for previous week can be expected
# Any time series analysis over the change in volumes, should omit these rows

#Remove NA values
dow = na.omit(dow)
sp500 = na.omit(sp500)

### Plot of percent price change over time
dow %>%
  ggplot(aes(x = date, y = percent_change_price, group = stock, color = stock)) +
  geom_line()

#Correlation Plot
corrplot::corrplot(cor(dow[,-c(1:3)]), method = c("number")) #Quite a few variables with high correlation 

#Multi Collinearity
lin.model = lm(percent_change_next_weeks_price ~ . , data = dow)
vif(lin.model) #Remove close

lin.model2 = lm(percent_change_next_weeks_price ~ . -close , data = dow) 
vif(lin.model2) #Remove high

lin.model3 = lm(percent_change_next_weeks_price ~ . -close -high , data = dow) 
vif(lin.model3) #Remove next_weeks_open

lin.model4 = lm(percent_change_next_weeks_price ~ . -close -high -next_weeks_open, data = dow) 
vif(lin.model4) #Remove low

lin.model5 = lm(percent_change_next_weeks_price ~ . -close -high -next_weeks_open -low, data = dow) 
vif(lin.model5) #Remove open

lin.model5 = lm(percent_change_next_weeks_price ~ . -close -high -next_weeks_open -low -open, data = dow) 
vif(lin.model5) #Remove next_weeks_close

lin.model6 = lm(percent_change_next_weeks_price ~ . -close -high -next_weeks_open -low -open -next_weeks_close, data = dow) 
vif(lin.model6) #Remove percent_return_next_dividend

lin.model7 = lm(percent_change_next_weeks_price ~ . -close -high -next_weeks_open -low -open -next_weeks_close -percent_return_next_dividend, data = dow) 
vif(lin.model7) #all vifs<5

#Remove variables from data set
dow = subset(dow, select = -c(close, high, next_weeks_open, low, open, next_weeks_close, percent_return_next_dividend))

#Box plots of variables
grid.arrange(ggplot(dow, aes(volume)) + geom_boxplot(),
             ggplot(dow, aes(percent_change_price)) + geom_boxplot(),
             ggplot(dow, aes(percent_change_volume_over_last_wk)) + geom_boxplot(),
             ggplot(dow, aes(previous_weeks_volume)) + geom_boxplot(),
             ggplot(dow, aes(percent_change_next_weeks_price)) + geom_boxplot(),
             ggplot(dow, aes(days_to_next_dividend)) + geom_boxplot(),
             ncol = 2,
             bottom = 'Figure X.X: Boxplots of Numerical Values')

#Numerical data is pretty skewed. Likely worth normalizing. 
#install.packages("bestNormalize")

dow_norm = lapply(dow[,c(4:9)], yeojohnson) #normalize numeric data 
#scale function just changed the scale, it was still skewed. log and sqrt gave errors to ended up with "yeojohnson." Appears to work well! 
dow_norm1 = cbind(dow[,1:3], volume = dow_norm$volume$x.t, percent_change_price = dow_norm$percent_change_price$x.t, 
                  percent_change_volume_over_last_wk = dow_norm$percent_change_volume_over_last_wk$x.t, 
                  previous_weeks_volume = dow_norm$previous_weeks_volume$x.t, percent_change_next_weeks_price = 
                    dow_norm$percent_change_next_weeks_price$x.t, days_to_next_dividend = dow_norm$days_to_next_dividend$x.t ) #combine factor data with normalized data and leaving out the index column ("observation")

grid.arrange(ggplot(dow_norm1, aes(volume)) + geom_boxplot(),
             ggplot(dow_norm1, aes(percent_change_price)) + geom_boxplot(),
             ggplot(dow_norm1, aes(percent_change_volume_over_last_wk)) + geom_boxplot(),
             ggplot(dow_norm1, aes(previous_weeks_volume)) + geom_boxplot(),
             ggplot(dow_norm1, aes(percent_change_next_weeks_price)) + geom_boxplot(),
             ggplot(dow_norm1, aes(days_to_next_dividend)) + geom_boxplot(),
             ncol = 2,
             bottom = 'Figure X.X: Boxplots of Numerical Values')

#Split into train & test data sets 
#Per the case study, quarter 1 will be used as the training data set and quarter 2 will be test data set 
dow_train = dow_norm1[dow_norm1$quarter==1,]
dow_test = dow_norm1[dow_norm1$quarter==2,]

#FOR LOOP to run each model on different stocks 
stocks <- unique(dow$stock) #create vector of stock names 

rmse = function(predicted, actual) {
  sqrt(mean((predicted-actual)^2))
}

results = data.frame(Stock = character(), RMSE = numeric(), stringsAsFactors = FALSE) #create an empty data frame to fill,
for (stock in stocks) {
  #Filter data sets based on stock
  dow_train_stock = dow_train[dow_train$stock == stock,] 
  dow_train_stock = dow_train_stock %>% dplyr::select(-stock)
  dow_test_stock = dow_test[dow_test$stock == stock,]
  dow_test_stock = dow_test_stock %>%  dplyr::select(-stock)
  #Fit models on training data set
  model = glm(percent_change_next_weeks_price ~ . , data = dow_train_stock) 
  
  #Predict on test data
  dow_test_stock$PredPercentChange = predict(model, newdata = dow_test_stock, type = "response")
  
  #Performance metric
  rmse_value = rmse(dow_test_stock$PredPercentChange, dow_test_stock$percent_change_next_weeks_price)
  
  #Store performance in data table
  results = rbind(results, data.frame(Stock = stock, RMSE = rmse_value)) #add onto previous data or empty df.
}

print(results)

# CAPM

capm_results = data.frame(Stock = character(), Beta_coef = numeric(), stringsAsFactors = FALSE) #create an empty data frame to fill
for (i in unique(dow$stock)) {
  # filter data for individual stocks
  dow_stock = dow %>%
                  dplyr::filter(dow$stock == i) %>%
                  dplyr::select(percent_change_next_weeks_price)
  sp500_data = sp500 %>%
                  dplyr::select(percent_change_next_weeks_price)
  capm_data = cbind(sp500_data, dow_stock)
  colnames(capm_data) = c("SP500", "Stock")
  
  # Model
  lm_model = lm(Stock ~ SP500, data = as.data.frame(capm_data))
  beta_coef <- summary(lm_model)$coefficients[2, 1]
  
  # Store beta coefficients in data table
  capm_results = capm_results %>%
                            rbind(data.frame(Stock = i, Beta_coef = beta_coef))
}

capm_results = capm_results %>%
                            arrange(desc(Beta_coef))

print(capm_results)

## Logistic Model
log.model8 = glm(percent_change_next_weeks_price ~ . , data = dow_train) 
summary(log.model8)

#Predictions 
dow_test$PredPercentChange = predict.glm(log.model8, newdata = dow_test, type = "response")

actual_vs_pred = data.frame(Actual = dow_test$percent_change_next_weeks_price, Predicted = dow_test$PredPercentChange)

rmse = sqrt(mean((actual_vs_pred$Actual - actual_vs_pred$Predicted)^2))
print(paste("RMSE:", rmse))


# Set a threshold to classify the predicted percentage change
# if the predicted change is greater than 0, classify it as an increase
dow_test$Direction = ifelse(dow_test$percent_change_next_weeks_price > 0, 1, 0)
dow_test$Direction = as.factor(dow_test$Direction)
dow_test$PredDirection = ifelse(dow_test$PredPercentChange > 0, 1, 0)
dow_test$PredDirection = as.factor(dow_test$PredDirection)

# Confusion Matrix to compare actual vs predicted directional movements
conf_matrix_glm = caret::confusionMatrix(dow_test$PredDirection, dow_test$Direction, positive = "1")
print(conf_matrix_glm)

# accuracy, sensitivity, and specificity
accuracy = conf_matrix_glm$overall["Accuracy"]
sensitivity = conf_matrix_glm$byClass["Sensitivity"]
specificity = conf_matrix_glm$byClass["Specificity"]

print(paste("Accuracy:", accuracy))
print(paste("Sensitivity:", sensitivity))
print(paste("Specificity:", specificity))

### LDA Model ###

# LDA Feature Selection
rand_f.model = randomForest::randomForest(percent_change_next_weeks_price ~ ., data = dow_train)

varImpPlot(rand_f.model,
           sort = T,
           n.var = 10,
           main = "Figure X. Variable Important plot")

rand_f.model2 = randomForest::randomForest(percent_change_next_weeks_price ~ . -quarter -stock, data = dow_train)
#removed quarter (least) and stock so that all features were above 20- this grouping was highest

varImpPlot(rand_f.model2,
           sort = T,
           n.var = 10,
           main = "Figure X. Variable Important plot")

# Make predictions on the test data
rf_pred = predict(rand_f.model2, dow_test)
print(rf_pred)

dow_test$Direction_rf = ifelse(dow_test$percent_change_next_weeks_price == 0, 1, 0)
dow_test$Direction_rf = as.factor(dow_test$Direction_rf)

# Confusion Matrix to compare actual vs predicted direction 
conf_matrix_rf = caret::confusionMatrix(rf_pred, dow_test$Direction, positive = "1")
print(conf_matrix_rf)



# SVR Model with Grid Search for Hyper parameter Tuning
# Define tuning grid for cost and gamma (for radial kernel)
tune_grid <- expand.grid(C = c(0.01, 0.1, 1, 10, 100), 
                         sigma = c(0.001, 0.01, 0.1, 1))

# Tuning Radial Basis Function (RBF) kernel SVM
set.seed(123)
svr_tune <- tune(svm, percent_change_next_weeks_price ~ ., data = dow_train,
                 type = "eps-regression",
                 kernel = "radial", 
                 ranges = list(cost = tune_grid$C, gamma = tune_grid$sigma),
                 scale = FALSE)  # Don't scale inside the SVM function, we've already done that

# Best model based on cross-validation
best_model <- svr_tune$best.model
print(svr_tune)

# Test the tuned SVM model
pred_svr <- predict(best_model, newdata = dow_test)

#RMSE
rmse = sqrt(mean((dow_test$percent_change_next_weeks_price - pred_svr)^2))
print(paste("RMSE:", rmse))

# Testing Other Kernels
# Linear Kernel
set.seed(123)
svr_linear <- svm(percent_change_next_weeks_price ~ ., data = dow_train,
                  type = "eps-regression",
                  kernel = "linear", 
                  cost = best_model$cost, 
                  scale = FALSE)
pred_linear <- predict(svr_linear, newdata = dow_test)
#RMSE
rmse = sqrt(mean((dow_test$percent_change_next_weeks_price - pred_linear)^2))
print(paste("RMSE:", rmse))

# Polynomial Kernel (degree 3)
set.seed(123)
svr_poly <- svm(percent_change_next_weeks_price ~ ., data = dow_train,
                type = "eps-regression", 
                kernel = "polynomial", 
                cost = best_model$cost, 
                degree = 3, 
                scale = FALSE)
pred_poly <- predict(svr_poly, newdata = dow_test)
#RMSE
rmse = sqrt(mean((dow_test$percent_change_next_weeks_price - pred_poly)^2))
print(paste("RMSE:", rmse))


### Decision Tree/RF ####

# Set seed for reproducibility
set.seed(123)

# Build a regression tree model using rpart with method = "anova"
decision_tree_model <- rpart(percent_change_next_weeks_price ~ ., data = dow_train, method = "anova")

# Predict on the test set with continuous values
dt_predictions <- predict(decision_tree_model, dow_test)

# Evaluate model performance with metrics like RMSE, MAE, etc., since this is regression
rmse <- sqrt(mean((dt_predictions - dow_test$percent_change_next_weeks_price)^2))
mae <- mean(abs(dt_predictions - dow_test$percent_change_next_weeks_price))

# Train the Decision Tree model with default parameters
decision_tree_model <- rpart(percent_change_next_weeks_price ~ ., data = dow_train, method = "anova")

# Hyperparameter Tuning for the Decision Tree (finding best complexity parameter 'cp')
best_cp <- decision_tree_model$cptable[which.min(decision_tree_model$cptable[,"xerror"]), "CP"]
tuned_decision_tree_model <- rpart(percent_change_next_weeks_price ~ ., data = dow_train, method = "anova",
                                   control = rpart.control(cp = best_cp))

# Predictions with the tuned Decision Tree model
tuned_dt_predictions <- predict(tuned_decision_tree_model, dow_test)

# Evaluate tuned Decision Tree performance
tuned_dt_rmse <- sqrt(mean((tuned_dt_predictions - dow_test$percent_change_next_weeks_price)^2))
tuned_dt_mae <- mean(abs(tuned_dt_predictions - dow_test$percent_change_next_weeks_price))

# Random Forest Model
rf_model <- randomForest(percent_change_next_weeks_price ~ ., data = dow_train, ntree = 100)

# Predictions with the Random Forest model
rf_predictions <- predict(rf_model, dow_test)

# Evaluate Random Forest performance
rf_rmse <- sqrt(mean((rf_predictions - dow_test$percent_change_next_weeks_price)^2))
rf_mae <- mean(abs(rf_predictions - dow_test$percent_change_next_weeks_price))

# Print Random Forest, tuned Decision Tree, and Decision Tree performance metrics
print(paste("Decision Tree Model RMSE:", round(rmse, 2)))
print(paste("Decision Tree Model MAE:", round(mae, 2)))
print(paste("Tuned Decision Tree RMSE:", round(tuned_dt_rmse, 2)))
print(paste("Tuned Decision Tree MAE:", round(tuned_dt_mae, 2)))
print(paste("Random Forest RMSE:", round(rf_rmse, 2)))
print(paste("Random Forest MAE:", round(rf_mae, 2)))

