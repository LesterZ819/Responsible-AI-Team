library(ggplot2)
library(dplyr)
library(reshape2)

IBM <- read.csv(url("https://raw.githubusercontent.com/LesterZ819/Responsible-AI-Team/master/UF/data/IBM.csv"))
attach(IBM)

library(fairness)


#Demographic parity
#Demographic parity is achieved if the absolute number of positive predictions in the subgroups are close to each other. This measure does not take true class into consideration, only positive predictions.
#Formula: (TP + FP)
dem=dem_parity(data         = IBM, 
             outcome      = 'Attrition',
             group        = 'Gender',
             probs        = 'prob', 
             preds_levels = c('No', 'Yes'), 
             cutoff       = 0.5, 
             base         = 'Male')

dem=dem_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'MaritalStatus',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Single')

dem$Metric_plot
dem$Metric

#Proportional parity
#Proportional parity is very similar to Demographic parity. Proportional parity is achieved if the proportion of positive predictions in the subgroups are close to each other. This measure does not take true class into consideration, only positive predictions.
#Formula: (TP + FP) / (TP + FP + TN + FN)
pp=prop_parity(data         = IBM, 
            outcome      = 'Attrition',
            group        = 'Gender',
            probs        = 'prob', 
            preds_levels = c('No', 'Yes'), 
            cutoff       = 0.5, 
            base         = 'Male')

pp=prop_parity(data         = IBM, 
                    outcome      = 'Attrition',
                    group        = 'MaritalStatus',
                    probs        = 'prob', 
                    preds_levels = c('No', 'Yes'), 
                    cutoff       = 0.5, 
                    base         = 'Single')

pp$Metric_plot
pp$Metric

#Equalized odds
#Equalized odds are achieved if the sensitivities (true positives divided by all positives) in the subgroups are close to each other.
#Formula: TP / (TP + FN)
eo = equal_odds(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

eo=equal_odds(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'MaritalStatus',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Single')

eo$Metric_plot
eo$Metric

#Predictive rate parity
#Predictive rate parity is achieved if the precisions or positive predictive values (true positives divided by all predicted positive) in the subgroups are close to each other.
#Formula: TP / (TP + FP)
pr=pred_rate_parity(data         = IBM, 
                 outcome      = 'Attrition',
                 group        = 'Gender',
                 probs        = 'prob', 
                 preds_levels = c('No', 'Yes'), 
                 cutoff       = 0.5, 
                 base         = 'Male')

pr =pred_rate_parity(data         = IBM, 
                      outcome      = 'Attrition',
                      group        = 'MaritalStatus',
                      probs        = 'prob', 
                      preds_levels = c('No', 'Yes'), 
                      cutoff       = 0.5, 
                      base         = 'Single')

pr$Metric_plot
pr$Metric

#Accuracy parity
#Accuracy parity is achieved if the accuracies (all accurately classified divided by all predictions) in the subgroups are close to each other.
#Formula: (TP + TN) / (TP + FP + TN + FN)
a=acc_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

a=acc_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'MaritalStatus',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Single')

a$Metric
a$Metric_plot

#False negative rate parity
#False negative rate parity is achieved if the false negative rates (division of false negatives with all positives) in the subgroups are close to each other.
#Formula: FN / (TP + FN)
fnr=fnr_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

fnr_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'MaritalStatus',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Single')

fnr$Metric
fnr$Metric_plot

#False positive rate parity
#False positive rate parity is achieved if the false positive rates (division of false positives with all negatives) in the subgroups are close to each other.
#Formula: FP / (TN + FP)
fpr=fpr_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

fpr=fpr_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'MaritalStatus',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Single')

fpr$Metric_plot
fpr$Metric
#Negative predictive value parity
#Negative predictive value parity is achieved if the negative predictive values (division of true negatives with all predicted negatives) in the subgroups are close to each other. This function can be considered the 'inverse' of Predictive rate parity.
#Formula: TN / (TN + FN)
npv=npv_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

npv=npv_parity(data         = IBM, 
               outcome      = 'Attrition',
               group        = 'MaritalStatus',
               probs        = 'prob', 
               preds_levels = c('No', 'Yes'), 
               cutoff       = 0.5, 
               base         = 'Single')

npv$Metric
npv$Metric_plot

#Specificity parity
#Specificity parity is achieved if the specificities (division of true negatives with all negatives) in the subgroups are close to each other. This function can be considered the 'inverse' of Equalized odds.
#Formula: TN / (TN + FP)
sp=spec_parity(data         = IBM, 
            outcome      = 'Attrition',
            group        = 'Gender',
            probs        = 'prob', 
            preds_levels = c('No', 'Yes'), 
            cutoff       = 0.5, 
            base         = 'Male')

sp=spec_parity(data         = IBM, 
            outcome      = 'Attrition',
            group        = 'MaritalStatus',
            probs        = 'prob', 
            preds_levels = c('No', 'Yes'), 
            cutoff       = 0.5, 
            base         = 'Single')
sp$Metric
sp$Metric_plot

#ROC AUC comparison
#This function calculates ROC AUC and visualizes ROC curves for all subgroups. Note that probabilities must be defined for this function. Also, as ROC evaluates all possible cutoffs, the cutoff argument is excluded from this function.
roc=roc_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           base         = 'Male')

roc=roc_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'MaritalStatus',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           base         = 'Single')

roc$ROCAUC_plot

#Matthews correlation coefficient comparison
#The Matthews correlation coefficient takes all 4 classes of the confusion matrix into consideration. According to some, it is the single most powerful metric in binary classification problems, especially for data with class imbalances.
#Formula: (TP?TN-FP?FN)/???((TP+FP)?(TP+FN)?(TN+FP)?(TN+FN))
mcc=mcc_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

mcc=mcc_parity(data         = IBM, 
               outcome      = 'Attrition',
               group        = 'MaritalStatus',
               probs        = 'prob', 
               preds_levels = c('No', 'Yes'), 
               cutoff       = 0.5, 
               base         = 'Single')

mcc$Metric_plot