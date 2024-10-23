pacman::p_load(caret, lattice, tidyverse, gam, logistf, MASS, car, corrplot, gridExtra, ROCR, RCurl, randomForest, readr, readxl, e1071, klaR)

##### Data Set ######
dow_raw = as.data.frame(read.csv(text = getURL('https://raw.githubusercontent.com/btj5z2/DA6813/main/dow_jones_index.data'), header = TRUE))

##### Copy of Data Set for Model ######
dow = dow_raw

### Review Details of Data Set ###
str(dow)

# Many numeric values were read in as strings
# Convert these values to numeric data types

num_vars = c('open', 'high', 'low', 'close', 'next_weeks_open', 'next_weeks_close')
dow[num_vars] = lapply(dow[num_vars], gsub, pattern = '[\\$,]', replacement = '')
dow[num_vars] = lapply(dow[num_vars], as.numeric)

# Convert 'date' column to date type

dow$date = as.Date(dow$date, '%m/%d/%Y')

### Review column details to validate changes ###
str(dow)

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

### Plot of percent price change over time
dow %>%
  ggplot(aes(x = date, y = percent_change_price, group = stock, color = stock)) +
  geom_line()

#Correlation Plot
corrplot::corrplot(cor(dow[,-c(2:3)]), method = c("number")) #Quite a few variables with high correlation 

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
library(bestNormalize)
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
dow_train = dow_norm[dow_norm$quarter==1,]
dow_test = dow_norm[dow_norm$quarter==2,]


