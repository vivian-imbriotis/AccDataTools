dat <- read.csv("C:/Users/viviani/Desktop/test.csv")
a <- dat[dat$ROI_ID == rois[3],]
train <- a[1:10000,]
train_no_trial = train[train$trial_factor== -999,]

# We might only want to train the model outside of trials as presumably more of the
# variances is due to things other than licking in that phase. *might*

mdl.full <-  lm(dF_on_F ~ lick_factor, data = train) #Train on all training data
mdl.no_trial <- lm(dF_on_F ~ lick_factor, data = train_no_trial) #Train on inter-trial data

# Only fair to test the model during licks, otherwise models will always look worse when tested
# against data sets with less licks.

test <- a[-(1:10000),]
test <- test[test$lick_factor != -999,]
test_no_trial = test[test$trial_factor == -999,]

predict.full <- predict(mdl.full, newdata = test) #predict all test data with
predict.full.no_trial <- predict(mdl.full, newdata = test_no_trial)
predict.no_trial.no_trial <- predict(mdl.no_trial, newdata = test_no_trial)

#Calculate Mean Squared Prediction Error
mspe.full = sqrt(mean((test$dF_on_F - predict.full)^2))
mspe.full.no_trial = sqrt(mean( (test_no_trial$dF_on_F - predict.full.no_trial)^2 ))
mspe.no_trial.no_trial = sqrt(mean( (test_no_trial$dF_on_F - predict.no_trial.no_trial)^2  ))

#Lets see this graphically
#Real Against Predicted
plot(test_no_trial$dF_on_F, predict.no_trial.no_trial, xlab = "Real Data", ylab = "Predicted Value")
abline(lm(test_no_trial$dF_on_F ~predict.no_trial.no_trial)) #at least in my data, line is close to slop = 1.

#Plot them both against time
plot(test_no_trial$dF_on_F, type='l')
lines(1:length(predict.no_trial.no_trial), predict.no_trial.no_trial, col = "red")

# Finally, if our prediction was really good, then there should be no relationship
# in the residual of the model against licking

resid.mdl <- lm(test_no_trial$dF_on_F - predict.no_trial.no_trial ~ test_no_trial$lick_factor)
summary(resid.mdl)