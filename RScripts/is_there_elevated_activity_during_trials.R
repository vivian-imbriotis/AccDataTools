dat <- read.csv("C:/Users/viviani/Desktop/full_datasets_for_analysis/left_only_high_contrast.csv")


get_lm_pvalue <- function (modelobject) {
  if (class(modelobject) != "lm") stop("Not an object of class 'lm' ")
  f <- summary(modelobject)$fstatistic
  p <- pf(f[1],f[2],f[3],lower.tail=F)
  attributes(p) <- NULL
  return(p)
}



rois            <- unique(dat$ROI_ID)
n               <- length(rois)
model_pvals     <- numeric(n)
model_estimates <- numeric(n)

for (i in 1:n){
  roi <- rois[i]
  roi.dat <- dat[dat$ROI_ID == roi,]
  during_trial_model <- lm(dF_on_F ~ I(trial_factor>0),
                           dat = roi.dat)
  model_pvals[i]    <- get_lm_pvalue(during_trial_model)
  model_estimates[i]<- during_trial_model$coefficients[[2]]
}

model_pvals_a <- p.adjust(model_pvals,'fdr')
sig_model_estimates <- model_estimates[model_pvals_a<0.05]
sum(sig_model_estimates>0)/length(sig_model_estimates)
write.csv(data.frame(model_estimates,model_pvals_a),"during_trials.csv")
