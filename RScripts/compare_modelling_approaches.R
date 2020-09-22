source_file <- "C:/Users/viviani/Desktop/single_experiments_for_testing/2016-11-05_03_CFEB029.csv"

set.seed(123456789) #Non


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
  df.tone <- dat$dF_on_F[dat$trial_component == 'Tone']
  df.tone <- df.tone[1:num_trials*5]
  df.tone.matr <- matrix(df.tone,nrow=num_trials,ncol=5,byrow=TRUE)
  mean.df.tone <- rowSums(df.tone.matr,na.rm=T)/5
  
  df.stim <- dat$dF_on_F[dat$trial_component=='Stim']
  df.stim <- df.stim[1:num_trials*10]
  df.stim.matr = matrix(df.stim,nrow=num_trials,ncol=10,byrow=TRUE)
  mean.df.stim <- rowSums(df.stim.matr,na.rm=T)/10
  
  df.resp <- dat$dF_on_F[dat$trial_component=='Resp']
  df.resp <- df.resp[1:num_trials*10]
  df.resp.matr <- matrix(df.resp,nrow=num_trials,ncol=10,byrow=TRUE)
  mean.df.resp <- rowSums(df.resp.matr,na.rm=T)/10
  
  result_idx = seq(1,3*num_trials,3)

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
dat <- dat[!is.na(dat$dF_on_F),]
dat$correct <- as.factor(dat$correct)
dat$go <- as.factor(dat$go)

rois <- unique(dat$ROI_ID)
n <- length(rois)

full.model.pvals <- numeric(n)
full.model.stats <- numeric(n)
full.model.rsqds <- numeric(n)
full.model.shapiro.stat <- numeric(n)
full.model.shapiro.pval <- numeric(n)

collapsed.model.pvals <- numeric(n)
collapsed.model.stats <- numeric(n)
collapsed.model.rsqds <- numeric(n)
collapsed.model.shapiro.stat <- numeric(n)
collapsed.model.shapiro.pval <- numeric(n)


for (i in 1:n){
  roi <- rois[[i]]
  roidat <- dat[dat$ROI_ID==roi,]
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
  
  full.model.residuals <- full.model$fitted.values - roidat[roidat$trial_factor!=-999,]$dF_on_F
  full.model.shapiro.test <- shapiro.test(sample(full.model.residuals,
                                                 min(100,length(full.model.residuals))))
  full.model.shapiro.stat[i] <- full.model.shapiro.test$statistic
  full.model.shapiro.pval[i] <-full.model.shapiro.test$p.value
  
  collapsed.model.test<- lmtest::dwtest(collapsed.model)
  collapsed.model.pvals[i] <- collapsed.model.test$p.value
  collapsed.model.stats[i] <- collapsed.model.test$statistic
  collapsed.model.rsqds[i] <- summary(collapsed.model)$adj.r.squared

  collapsed.model.residuals <- (collapsed.model$fitted.values - 
                                    collapsed.after.licking.subtraction$mean.dF)
  collapsed.model.shapiro.test <- shapiro.test(sample(
                                      collapsed.model.residuals,
                                      min(100,length(collapsed.model.residuals))))
  collapsed.model.shapiro.stat[i] <- collapsed.model.shapiro.test$statistic
  collapsed.model.shapiro.pval[i] <- collapsed.model.shapiro.test$p.value
  
}


