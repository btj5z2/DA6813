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

### Plot of percent price change over time
dow %>%
  ggplot(aes(x = date, y = percent_change_price, group = stock, color = stock)) +
  geom_line()
