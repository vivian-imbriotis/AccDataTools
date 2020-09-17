source_file_left_only <- "C:/Users/viviani/Desktop/full_datasets_for_analysis/left_only_high_contrast.csv"
source_file_left_and_right <- "C:/Users/viviani/Desktop/full_datasets_for_analysis/both_sides_high_contrast.csv"
source_file_low_contrast <- "C:/Users/viviani/Desktop/full_datasets_for_analysis/low_contrast.csv"

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
  trials.go      <- dat$go[dat$trial_factor==1]
  trials.side    <- dat$side[dat$trial_factor==1]
  trials.contrast<- dat$contrast[dat$trial_factor==1]
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
  
  result$correct <- as.factor(rep(trials.correct,3))
  result$go      <- as.factor(rep(trials.go,3))
  result$side    <- as.factor(rep(trials.side,3))
  result$contrast<- as.factor(rep(trials.contrast,3))
  
  
  result <- na.omit(result)
  result$trial.segment <- as.factor(result$trial.segment)
  return(result)
}
###############################################
## ANALYSIS OF LEFT_ONLY (MONOCULAR) DATASET ##
###############################################
print("Beginning Analysis of Monocular Data...")
dat <- read.csv(source_file_left_only)
dat <- dat[!is.na(dat$dF_on_F),]
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
                                    +trial.segment:go,
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

anova_frame_pvals <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`Pr(>F)`))))
anova_frame_fvals <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`F value`))))
colnames(anova_frame_pvals) <- colnames(anova_frame_pvals) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste(x,"pvalue",sep="."))
colnames(anova_frame_fvals) <- colnames(anova_frame_fvals) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste(x,"fvalue",sep="."))
#Drop the residuals columns from the ANOVA output matrix
anova_frame_pvals <- anova_frame_pvals[,1:3]
anova_frame_fvals <- anova_frame_fvals[,1:3]
anova_frame <- cbind(anova_frame_pvals,anova_frame_fvals)
write.csv(anova_frame,"left_only_collapsed_lm_anova_results.csv")

#######################################################
## ANALYSIS OF BINOCULAR, HIGH-CONTRAST STIM DATASET ##
#######################################################
print("Beginning analysis of binocular high contrast data")
dat <- read.csv(source_file_left_and_right)
dat <- dat[!is.na(dat$dF_on_F),]
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
                                   +trial.segment:go + trial.segment:side,
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

anova_frame_pvals <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`Pr(>F)`))))
anova_frame_fvals <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`F value`))))
colnames(anova_frame_pvals) <- colnames(anova_frame_pvals) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste(x,"pvalue",sep="."))
colnames(anova_frame_fvals) <- colnames(anova_frame_fvals) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste(x,"fvalue",sep="."))
#Drop the residuals columns from the ANOVA output matrix
anova_frame_pvals <- anova_frame_pvals[,1:4]
anova_frame_fvals <- anova_frame_fvals[,1:4]
anova_frame <- cbind(anova_frame_pvals,anova_frame_fvals)
write.csv(anova_frame,"both_sides_collapsed_lm_anova_results.csv")

#######################################################
## ANALYSIS OF BINOCULAR, LOW-CONTRAST STIM DATASET  ##
#######################################################
print("Beginning analysis of low-contrast data")
dat <- read.csv(source_file_low_contrast)
dat <- dat[!is.na(dat$dF_on_F),]
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
                                   +trial.segment:go + trial.segment:side
                                   +trial.segment:correct:contrast,
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

anova_frame_pvals <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`Pr(>F)`))))
anova_frame_fvals <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`F value`))))
colnames(anova_frame_pvals) <- colnames(anova_frame_pvals) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste(x,"pvalue",sep="."))
colnames(anova_frame_fvals) <- colnames(anova_frame_fvals) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste(x,"fvalue",sep="."))
#Drop the residuals columns from the ANOVA output matrix
anova_frame_pvals <- anova_frame_pvals[,1:5]
anova_frame_fvals <- anova_frame_fvals[,1:5]
anova_frame <- cbind(anova_frame_pvals,anova_frame_fvals)
write.csv(anova_frame,"both_sides_collapsed_lm_anova_results.csv")
