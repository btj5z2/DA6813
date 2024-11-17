# Libraries
pacman::p_load(caret, lattice, tidyverse, gam, logistf, MASS, 
               car, corrplot, gridExtra, ROCR, RCurl, randomForest, 
               readr, readxl, e1071, klaR, bestNormalize, rpart, lubridate,
               tseries, quantmod, knitr, SMCRM)

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

str(crm)

### Duration sub-group

crm_dur = crm %>%
  filter(acquisition == 1) %>% # filter out unacquired customers
  dplyr::select(-c(acquisition)) # drop acquisition

### Check for NA

which(is.na(crm))
### No NA found

### Viz features - Acquisition

grid.arrange(
  ggplot(crm, aes(acquisition, duration)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, profit)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, acq_exp)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, ret_exp)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, freq)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, freq_sq)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, crossbuy)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, sow)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, revenue)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, employees)) + geom_boxplot(),
  ggplot(crm, aes(acquisition, after_stat(count))) + geom_bar(aes(fill = industry), position = 'dodge'),
  bottom = 'Figure X.X: Plots of predictor relationship with acquisition response'
)
#### duration, ret_exp, freq, freq_sq, crossbuy, and sow are perfect predictors
#### all of these features will only return a value if the customer is acquired
#### otherwise, these are 0

### Create acquisition data set with perfect predictors removed
crm_acq = crm %>%
              dplyr::select(-c(duration, ret_exp, freq, freq_sq, crossbuy, sow))

str(crm_acq)

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

### Viz response variables

#### acquisition
crm_acq %>%
  ggplot(aes(acquisition)) +
  geom_bar()
#### data is imbalanced

#### duration

crm_dur %>%
  ggplot(aes(duration)) +
  geom_histogram(bins = 30)

### Check for multicollinearity

#### Acquisition

##### Corr Plot

corrplot::corrplot(cor(crm_acq[,-c(1,6)]))

##### VIF

#### Duration

##### Corr Plot

corrplot::corrplot(cor(crm_dur[,-c(11)]))

##### VIF