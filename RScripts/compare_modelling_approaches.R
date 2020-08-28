source_file <- "C:/Users/viviani/Desktop/single_experiments_for_testing/2016-11-01_03_CFEB027.csv"
remove_inflated_values <- TRUE

set.seed(123456789)


require("lmtest")

get_lm_pvalue <- function (modelobject) {
  if (class(modelobject) != "lm") stop("Not an object of class 'lm' ")
  f <- summary(modelobject)$fstatistic
  p <- pf(f[1],f[2],f[3],lower.tail=F)
  attributes(p) <- NULL
  return(p)
}

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
  mean.df.tone <- colSums(matrix(df.tone,5),na.rm=T)/5
  
  df.stim <- dat$dF_on_F[(dat$trial_factor>=6 & dat$trial_factor)<16]
  df.stim <- df.stim[1:(10*(length(df.stim) %/% 10))]
  mean.df.stim <- colSums(matrix(df.stim,10),na.rm=T)/10
  
  df.resp <- dat$dF_on_F[dat$trial_factor>15]
  df.resp <- df.resp[1:(10*(length(df.resp) %/% 10))]/10
  mean.df.resp <- colSums(matrix(df.resp,10),na.rm=T)/10
  
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

dat <- read.csv(source_file)
dat$correct <- as.factor(dat$correct)
dat$go <- as.factor(dat$go)

rois <- unique(dat$ROI_ID)
n <- length(rois)

full.model.pvals <- numeric(n)
full.model.stats <- numeric(n)
full.model.rsqds <- numeric(n)
full.model.shapiro.stat <- numeric(n)

collapsed.model.pvals <- numeric(n)
collapsed.model.stats <- numeric(n)
collapsed.model.rsqds <- numeric(n)
collapsed.model.shapiro.stat <- numeric(n)


for (i in 1:n){
  roi <- rois[[i]]
  roidat <- dat[dat$ROI_ID==roi,]
  if(remove_inflated_values){
    inflated_val <- names(table(roidat$dF_on_F)[1])
    roidat[roidat$dF_on_F == inflated_val,"dF_on_F"] <- NA
  }
  outside_trials  <- roidat[roidat$trial_factor== -999,]
  licking.model <- lm(dF_on_F ~ lick_factor, 
                      data = outside_trials)
  
  if(get_lm_pvalue(licking.model)>0.05){
    licking_model <- lm(dF_on_F ~ 1, data = outside_trials)
  }
  
  licking.prediction <- predict(licking.model, newdata = roidat)
  roidat$residuals <- roidat$dF_on_F - licking.prediction
  residual.dat <- roidat
  residual.dat$dF_on_F <- roidat$dF_on_F - licking.prediction
  collapsed.after.licking.subtraction <- collapse.across.time(residual.dat)
  collapsed.model<- lm(mean.dF ~ trial.segment + trial.segment:correct
                                   +trial.segment:go + trial.segment:go:correct,
                                   data = collapsed.after.licking.subtraction)
  
  full.model <- lm(dF_on_F ~ as.factor(lick_factor) + as.factor(trial_factor) + 
                go:as.factor(trial_factor) + correct:as.factor(trial_factor),
              data = roidat[roidat$trial_factor!=-999,])
  
  full.model.test <- lmtest::dwtest(full.model,tol = 0)
  full.model.pvals[i] <- full.model.test$p.value
  full.model.stats[i] <- full.model.test$statistic
  full.model.rsqds[i] <- summary(full.model)$adj.r.squared
  
  full.model.residuals <- full.model$fitted.values - roidat$dF_on_F
  full.model.shapiro.stat[i] <- shapiro.test(sample(full.model.residuals,
                                                     min(3000,length(full.model.residuals))))$statistic
  
  collapsed.model.test<- lmtest::dwtest(collapsed.model)
  collapsed.model.pvals[i] <- collapsed.model.test$p.value
  collapsed.model.stats[i] <- collapsed.model.test$statistic
  collapsed.model.rsqds[i] <- summary(collapsed.model)$adj.r.squared

  collapsed.model.residuals <- (collapsed.model$fitted.values - 
                                    collapsed.after.licking.subtraction$mean.dF)
  collapsed.model.shapiro.stat[i] <- shapiro.test(sample(
                                                collapsed.model.residuals,
                                                min(3000,length(collapsed.model.residuals))))$statistic
  
}


