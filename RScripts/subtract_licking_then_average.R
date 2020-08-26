
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

licking_model_pvalues = numeric(length(rois))
licking_significant = 0
licking_insignificant = 0
summary_objects = vector(mode = "list", length= length(rois))
model_pvals = numeric(length(rois))
anovas = vector(mode = "list", length = length(rois))

for(i in 1:length(rois)){
  roi <- rois[i]
  subset <- dat[dat$ROI_ID==roi,]
  outside_trials  <- subset[subset$trial_factor== -999,]
  licking.model <- lm(dF_on_F ~ lick_factor, 
                      data = outside_trials)
  if(get_lm_pvalue(licking.model)>0.05){
    licking_insignificant = licking_insignificant + 1
    licking_model <- lm(dF_on_F ~ 1, data = outside_trials)
  }
  licking.prediction <- predict(licking.model, newdata = subset)
  subset$residuals <- subset$dF_on_F - licking.prediction
  residual.dat <- subset
  residual.dat$dF_on_F <- subset$dF_on_F - licking.prediction
  collapsed.after.licking.subtraction <- collapse.across.time(residual.dat)
  lm.with.licking.subtraction<- lm(mean.dF ~ trial.segment + trial.segment:correct
                                    +trial.segment:go + trial.segment:go:correct,
                                    data = collapsed.after.licking.subtraction)
  
  summary_objects[[i]] <- summary(lm.with.licking.subtraction)
  anovas[[i]] <- anova(lm.with.licking.subtraction)
  model_pvals[[i]] <- get_lm_pvalue(lm.with.licking.subtraction)
}

model_pvals     <- p.adjust(model_pvals, method = "fdr")
rsquareds       <- lapply(summary_objects, function(x) x$adj.r.squared)
coeffs          <- lapply(summary_objects, function(x) x$coefficients)
coeff_estimates <- data.frame(do.call(rbind, lapply(coeffs,function(x) x[,"Estimate"])))
coeff_pvals     <- data.frame(do.call(rbind, lapply(coeffs,function(x) x[,"Pr(>|t|)"])))

anova_frame <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`Pr(>F)`))))
colnames(anova_frame) <- row.names(anovas[[1]])
