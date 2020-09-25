require(effectsize)

source_file_left_only <- "C:/Users/viviani/Desktop/full_datasets_for_analysis/left_only_high_contrast.csv"
source_file_left_and_right <- "C:/Users/viviani/Desktop/full_datasets_for_analysis/both_sides_high_contrast.csv"
source_file_low_contrast <- "C:/Users/viviani/Desktop/full_datasets_for_analysis/low_contrast.csv"

#Helper function to get the overall p-value from a summary(lm) object
#(ie the result of the f-test that you see when you call summary(model))
get_lm_pvalue <- function (modelobject) {
  if (class(modelobject) != "lm") stop("Not an object of class 'lm' ")
  f <- summary(modelobject)$fstatistic
  p <- pf(f[1],f[2],f[3],lower.tail=F)
  attributes(p) <- NULL
  return(p)
}

#At times we will want to recoverably log-transform our data
log_transform <- function(values){
  minimum <- min(values)
  log(values - minimum + 10e-16)
}
inverse_log_transform <- function(values, minimum){
  exp(values) + minimum - 10e-16
}


#Function to perform fit the kernel-based model to each ROI and output a CSV
analyse_and_produce_csv_of_results <- function(source_file,destination_file,
                                               side_varying=FALSE,
                                               contrast_varying=FALSE){
  
  num_of_free_variables <- (3 + side_varying + contrast_varying)
  #Read in and clean the data
  cat("Reading in data...")
  dat <- read.csv(source_file)
  dat <- dat[!is.na(dat$dF_on_F),]
  dat$lick_factor  <- as.factor(dat$lick_factor)
  dat$trial_factor <- as.factor(dat$trial_factor)
  dat$correct      <- as.factor(dat$correct)
  dat$go           <- as.factor(dat$go)
  dat$side         <- as.factor(dat$side)
  dat$contrast     <- as.factor(dat$contrast)
  cat("done\nAnalysing...")
  
  #Construct vectors to hold the results for each ROI
  rois <- unique(dat$ROI_ID)
  summary_objects       <- vector(mode = "list", length= length(rois))
  model_pvals           <- numeric(length(rois))
  anovas                <- vector(mode = "list", length = length(rois))
  
  #For each ROI/bouton in the dataset...
  for(i in 1:length(rois)){
    roi     <- rois[i]
    subset  <- dat[dat$ROI_ID==roi,]
    minimum <- min(subset$dF_on_F)
    subset$logged.df <- log_transform(subset$dF_on_F)
    if(contrast_varying && side_varying){
      model<- lm(logged.df ~ lick_factor + 
                             trial_factor +
                             trial_factor:correct +
                             trial_factor:go+
                             trial_factor:side +
                             trial_factor:correct:contrast,
                             data = subset)
    }else if(side_varying){
      model<- lm(logged.df ~ lick_factor + 
                             trial_factor +
                             trial_factor:correct +
                             trial_factor:go+
                             trial_factor:side,
                             data = subset)
    }else{
      model<- lm(logged.df ~ lick_factor + 
                             trial_factor +
                             trial_factor:correct +
                             trial_factor:go,
                             data = subset)
    }
    summary_objects[[i]] <- summary(model)
    model_pvals[i]  <-get_lm_pvalue(modelobject = model)
    anovas[[i]] <- anova(model)
  }
  
  
  #Now construct a dataframe of all the relevant statistics for each ROI
  model_pvals_a   <- p.adjust(model_pvals, method = "fdr")                 #Overall model significance
  rsquareds       <- lapply(summary_objects, function(x) x$adj.r.squared)  #Overall adjusted R squared
  coeffs          <- lapply(summary_objects, function(x) x$coefficients)   
  coeff_estimates <- data.frame(do.call(rbind, lapply(coeffs,function(x) x[,"Estimate"])))     #Coefficient Estimates
  coeff_pvals     <- data.frame(do.call(rbind, lapply(coeffs,function(x) x[,"Pr(>|t|)"])))
  coeff_pvals_a   <- data.frame(lapply(coeff_pvals, FUN=function(x) p.adjust(x,method='fdr'))) #Coefficient pvalues
  
  
  #Name each column something sensible
  colnames(coeff_pvals)   <- sapply(colnames(coeff_pvals_a),FUN=function(x) paste('coefficient',x,"p.unadjusted",sep=" "))
  colnames(coeff_pvals_a)   <- sapply(colnames(coeff_pvals_a),FUN=function(x) paste('coefficient',x,"pvalue",sep=" "))
  colnames(coeff_estimates) <- sapply(colnames(coeff_estimates),FUN=function(x) paste('coefficient',x,"estimate",sep=" "))
  
  anova_frame_pvals_unadjusted <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`Pr(>F)`))))   #ANOVA p-values for each var  
  anova_frame_pvals <- data.frame(t(rbind(sapply(anovas,FUN=function(x) p.adjust(x$`Pr(>F)`,method='fdr')))))   #ANOVA p-values for each var
  anova_frame_fvals <- data.frame(t(rbind(sapply(anovas,FUN=function(x) x$`F value`))))  #ANOVA f-values
  #Finally, partial eta-squareds as a measure of effect size on ANOVA:
  anova_frame_etas  <- data.frame(t(rbind(sapply(anovas,FUN=function(x) effectsize::eta_squared(x)$Eta_Sq_partial))))
  
  colnames(anova_frame_pvals_unadjusted) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste('ANOVA',x,"p.unadjusted",sep=" "))
  colnames(anova_frame_pvals) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste('ANOVA',x,"pvalue",sep=" "))
  colnames(anova_frame_fvals) <- sapply(row.names(anovas[[1]]),FUN=function(x) paste('ANOVA',x,"fvalue",sep=" "))
  colnames(anova_frame_etas)  <- sapply(effectsize::eta_squared(anovas[[1]])$Parameter,
                                        FUN=function(x) paste('ANOVA',x,"partial_eta2",sep=" "))
  #Drop the residuals columns from the ANOVA output 
  anova_frame_pvals_unadjusted <- anova_frame_pvals_unadjusted[,1:num_of_free_variables]
  anova_frame_pvals <- anova_frame_pvals[,1:num_of_free_variables]
  anova_frame_fvals <- anova_frame_fvals[,1:num_of_free_variables]
  #Glue everything together and dump to CSV
  output_frame <- cbind(anova_frame_pvals_unadjusted,anova_frame_pvals,
                        anova_frame_fvals, anova_frame_etas, coeff_pvals, 
                        coeff_pvals_a,coeff_estimates)
  output_frame$`collapsed.model p.unadjsuted`     <- model_pvals
  output_frame$`collapsed.model pvalue`           <- model_pvals_a
  output_frame$`overall.model.adj.rsquared`       <- unlist(rsquareds)
  cat("done\nWriting CSV...")
  write.csv(output_frame,destination_file)
  cat("done\n")
  return(output_frame)
}


######################################################
##    ANALYSIS OF LEFT_ONLY (MONOCULAR) DATASET     ##
######################################################
print("Beginning analysis of monocular data...")
left_only_results <- analyse_and_produce_csv_of_results(source_file_left_only,
                                                        "results_left_only_fullkernel.csv")

#######################################################
## ANALYSIS OF BINOCULAR, HIGH-CONTRAST STIM DATASET ##
#######################################################
print("Beginning analysis of binocular high contrast data...")
binocular_high_con_results <- analyse_and_produce_csv_of_results(source_file_left_and_right,
                                                                 'results_binocular_fullkernel.csv',
                                                                 side_varying = TRUE)

#######################################################
## ANALYSIS OF BINOCULAR, LOW-CONTRAST STIM DATASET  ##
#######################################################
print("Beginning analysis of low-contrast data...")
binocular_low_con_results <- analyse_and_produce_csv_of_results(source_file_low_contrast,
                                                                'results_low_contrast_fullkernel.csv',
                                                                side_varying = TRUE,
                                                                contrast_varying = TRUE)
print("...done")