
collapse.across.time <- function(dat){
  num_trials <- sum(dat$trial_factor==1)
  trials.correct <- dat$correct[dat$trial_factor==1]
  trials.go <- dat$go[dat$trial_factor==1]
  trials.side <- dat$side[dat$trial_factor==1]
  result = data.frame(mean.dF = numeric(num_trials*3),
                      trial.segment = character(num_trials*3),
                      stringsAsFactors = FALSE)
  result$mean.dF <- NA
  result$trial.segment<-NA
  #We first need to select only the timepoints happening in the
  #portion of each trial we care about, then we need to sum 
  #together every K consecutive timepoints. In numpy i'd reshape
  #and sum along an axis...
  df.tone <- dat$dF_on_F[(dat$trial_factor!=(-999) & dat$trial_factor<6)]
  df.tone <- df.tone[1:(5*(length(df.tone) %/% 5))]
  mean.df.tone <- colSums(matrix(df.tone,5))/5
  
  df.stim <- dat$dF_on_F[(dat$trial_factor>=6 & dat$trial_factor)<16]
  df.stim <- df.stim[1:(10*(length(df.stim) %/% 10))]
  mean.df.stim <- colSums(matrix(df.stim,10))/10
  
  df.resp <- dat$dF_on_F[dat$trial_factor>15]
  df.resp <- df.resp[1:(11*(length(df.resp) %/% 11))]/11
  mean.df.resp <- colSums(matrix(df.resp,11))/11
  result_idx = seq(1,3*length(mean.df.tone),3)
  result$mean.dF[result_idx+0]       <- mean.df.tone
  result$trial.segment[result_idx+0] <- "Tone"
  result$mean.dF[result_idx+1]       <- mean.df.stim
  result$trial.segment[result_idx+1] <- "Stimulus"
  result$mean.dF[result_idx+2]       <- mean.df.resp
  result$trial.segment[result_idx+2] <- "Response"
  
  result$correct <- rep(trials.correct,3)
  result$go      <- rep(trials.go,3)
  result$side    <- rep(trials.side,3)
  
  
  result <- na.omit(result)
  result$trial.segment <- as.factor(result$trial.segment)
  return(result)
}


#Let's start out by getting a single ROI from a single recording.

dat <- read.csv("C:/Users/viviani/Desktop/test.csv")
rois <- unique(dat$ROI_ID)
dat <- dat[dat$ROI_ID == rois[2],]


# Let's fit a naive model first, which doesn't do the licking 
collapsed.before.licking.subtraction <- collapse.across.time(dat)
lm.no.licking.subtraction<- lm(mean.dF ~ trial.segment + trial.segment:correct
                                +trial.segment:go,
                                data = collapsed.before.licking.subtraction)

#Okay, now to fit a licking model so we can subtract off its predictions.
#We'll fit this model only on intertrial periods so we don't attribute
#eg Hit-trial-assosiated fluorescence to licking
outside_trials        <- dat[dat$trial_factor== -999,]
#Reserve some data for testing
training_cuttoff      <- floor(0.8*nrow(outside_trials))
licking.training.data <- outside_trials[1:training_cuttoff,] 
licking.testing.data  <- outside_trials[-(1:training_cuttoff),]

licking.model <- lm(dF_on_F ~ lick_factor, 
                    data = licking.training.data) 

# Before we go any further, was the licking model actually any good?
# We can check by making sure that (1) the model was significant, and 
# (2) we can no longer get a significant model of the residuals
# with licking as the independant variable (this idea courtesy of Bill).
licking.prediction.licktest   <- predict(licking.model, 
                                         newdata = licking.testing.data)
licking.testing.data$residuals <- (licking.testing.data$dF_on_F - 
                                     licking.prediction.licktest)
snd.order.licking.model<- lm(residuals ~ lick_factor,
                             data = licking.testing.data)

print("LICKING MODEL (SHOULD BE SIGNIFICANT):")
print(summary(licking.model))
cat("\n\nSECOND ORDER LICKING MODEL (SHOULD BE INSIGNIFICANT):")
print(summary(snd.order.licking.model))

# Okay, now let's get out model that fits residuals after licking
# subtraction based on trial features
licking.prediction.everywhere <- predict(licking.model, newdata = dat)
dat$residuals <- dat$dF_on_F - licking.prediction.everywhere
residual.dat <- dat
residual.dat$dF_on_F <- dat$dF_on_F - licking.prediction
collapsed.after.licking.subtraction <- collapse.across.time(residual.dat)

lm.with.licking.subtraction<- lm(mean.dF ~ trial.segment + trial.segment:correct
                                  +trial.segment:go,
                                  data = collapsed.after.licking.subtraction)

#Was our second model better?
cat("\n\nModel WITHOUT Licking Subtraction")
print(summary(lm.no.licking.subtraction))
cat("\n\nModel WITH Licking Subtraction")
print(summary(lm.with.licking.subtraction))

#We can't get an F-test off an anova because out models
#have the same number of degrees of freedom, but we
#can at least get the RSS.
cat("\n\nANOVA")
print(anova(lm.no.licking.subtraction,
            lm.with.licking.subtraction))