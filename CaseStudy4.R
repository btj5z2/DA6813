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

### Duration sub-group

crm_dur = crm %>%
  filter(acquisition == 1) %>% # filter out unacquired customers
  dplyr::select(-c(acquisition, freq, freq_sq, crossbuy)) 
  # drop columns fixed at 0 for unacquired customers

### Check for NA

which(is.na(crm))
### No NA found

### Chart response variables

#### acquisition
crm %>%
  ggplot(aes(acquisition)) +
  geom_bar()
#### data is imbalanced

#### duration

crm_dur %>%
  ggplot(aes(duration)) +
  geom_histogram(bins = 30)
