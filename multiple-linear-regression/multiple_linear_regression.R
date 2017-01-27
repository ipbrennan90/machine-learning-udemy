#Multiple linear regression
#Data pre process
#import dataset
dataset = read.csv('50_Startups.csv')


# encode categoricals
dataset$State = factor(dataset$State, 
                       levels = c('California', 'New York', 'Florida'),
                       labels = c(1,2,3))

library(caTools)
set.seed(123)

split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting Multiple Linear regression to training set
#this is how you say formula for profit = a linear combination of all the independent vars
regressor = lm(formula = Profit ~ ., data = training_set)

y_pred = predict(regressor, newdata = test_set)