dat <- read.csv("C:/Users/viviani/Desktop/test.csv")

model_peritrial <- lm(pupil_diameter ~ as.factor(peritrial_factor) + number_of_trials_seen:as.factor(peritrial_factor),
            data = dat)

model_intertrial <-lm(pupil_diameter ~ as.factor(trial_factor),
                      data = dat)


dat$trial_is_happening <- dat$trial_factor!=(-999)

rois <- unique(dat$ROI_ID)
n <- length(rois)
pvals = numeric(n)
coeffs = numeric(n)

for (i in 1:n){
  roi = rois[[i]]
  cat(sprintf("%d of %d       \r",i,n))
  subset = dat[dat$ROI_ID == roi,]
  model_trial <- lm(dF_on_F~trial_is_happening,
                                 data= subset)
  pvals[i] <- anova(model_trial)$`Pr(>F)`[1] #The pvalue
  coeffs[i]<- model_trial$coefficients[2]
}
pvals <- p.adjust(pvals, method = 'fdr')


  


