seed <- 123456789

#library(lme4)
library(merTools)
library(lmerTest)
require(emmeans)
dat <- read.csv("C:/Users/viviani/Desktop/pupil_unrolled.csv")
# dat$go <- dat$trial_type=="hit" | dat$trial_type == "miss"
# dat$correct <- dat$trial_type=="hit" | dat$trial_type=="cr"


base_model       <- lmer(pupil_diameter ~ trial_frame + (1|recording), 
                         data=dat, REML=FALSE)
# base_model_parabola <- lmer(pupil_diameter ~ trial_frame + I(trial_frame^2) + (1|recording),
#                             data = dat, REML = FALSE)
# base_model_factor <- lmer(pupil_diameter ~ as.factor(trial_frame)+(1|recording),
#                           data=dat,REML=FALSE)
trial_type_model    <- lmer(pupil_diameter ~ trial_frame*trial_type + (1|recording),
                         data=dat, REML=FALSE)
# go_correct_model <- lmer(pupil_diameter ~ go*correct*trial_type + (1|recording), 
#                          data=dat, REML=FALSE)
# trial_type_model_parabola<- lmer(pupil_diameter ~ I(trial_frame**2)*trial_type +trial_frame*trial_type + (1|recording), 
#                              data=dat, REML=FALSE)
# trial_type_model_factor<- lmer(pupil_diameter ~ as.factor(trial_frame)*trial_type - trial_type + (1|recording),
#                                  data=dat, REML=FALSE)
# go_correct_model_factor<- lmer(pupil_diameter ~ as.factor(trial_frame)*go*correct - go*correct + (1|recording), 
#                                data=dat, REML=FALSE)
# 
# anova_of_models  <- anova(trial_type_model, base_model)

trial_frame    <- numeric(26)
trial_frame[] <- 0:25
trial_type     <- numeric(26)
trial_type[]  <- "hit"
recording      <- numeric(26)
recording[]   <- dat$recording[1]
mock_hit <- data.frame(trial_frame,trial_type,recording)
mock_miss <- mock_hit
mock_miss$trial_type <- "miss"
mock_fa <- mock_hit
mock_fa$trial_type <- "fa"
mock_cr <- mock_hit
mock_cr$trial_type <- "cr"

save_predictions_to_csv <- function(model,path){
  intervals_hit <- predictInterval(model,
                                   mock_hit,
                                   which = "fixed",
                                   level = 0.5,
                                   n.sims=10000,
                                   seed = seed)
  intervals_hit$trial_type <- "hit"
  
  intervals_miss <- predictInterval(model,
                                   mock_miss,
                                   which = "fixed",
                                   level = 0.5,
                                   n.sims=10000,
                                   seed = seed)
  intervals_miss$trial_type <- "miss"
  
  intervals_cr <- predictInterval(model,
                                   mock_cr,
                                   which = "fixed",
                                   level = 0.5,
                                   n.sims=10000,
                                   seed = seed)
  intervals_cr$trial_type<-"cr"
  
  intervals_fa <- predictInterval(model,
                                   mock_fa,
                                   which = "fixed",
                                   level = 0.5,
                                   n.sims=10000,
                                   seed = seed)
  intervals_fa$trial_type <- "fa"
  out <- rbind(intervals_hit,intervals_miss,intervals_cr,intervals_fa)
  write.csv(out, 
            path)
}

# save_predictions_to_csv(trial_type_model_parabola,
#                         "C:/Users/viviani/Desktop/parabolic_mixed_linear_model_pupil_prediction.csv"
#                         )
base_model       <- lmer(pupil_diameter ~ trial_frame + (1|recording), 
                         data=dat, REML=FALSE)
trial_type_model    <- lmer(pupil_diameter ~ trial_frame*trial_type + (1|recording),
                            data=dat, REML=FALSE)
print(anova(base_model,trial_type_model))
var <- emmeans::emtrends(trial_type_model, pairwise ~ trial_type, var = "trial_frame")
posthoc <- summary(var)$contrasts
cat("\n\n")
print(posthoc)
# posthoc$p.value <- p.adjust(posthoc$p.value, method = "bonferroni")
