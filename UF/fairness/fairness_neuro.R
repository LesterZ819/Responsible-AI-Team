library(fairness)
#Demographic parity
#Demographic parity is achieved if the absolute number of positive predictions in the subgroups are close to each other. This measure does not take true class into consideration, only positive predictions.
#Formula: (TP + FP)
dem_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

#Proportional parity
#Proportional parity is very similar to Demographic parity. Proportional parity is achieved if the proportion of positive predictions in the subgroups are close to each other. This measure does not take true class into consideration, only positive predictions.
#Formula: (TP + FP) / (TP + FP + TN + FN)
prop_parity(data         = IBM, 
            outcome      = 'Attrition',
            group        = 'Gender',
            probs        = 'prob', 
            preds_levels = c('No', 'Yes'), 
            cutoff       = 0.5, 
            base         = 'Male')

#Equalized odds
#Equalized odds are achieved if the sensitivities (true positives divided by all positives) in the subgroups are close to each other.
#Formula: TP / (TP + FN)
equal_odds(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

#Predictive rate parity
#Predictive rate parity is achieved if the precisions or positive predictive values (true positives divided by all predicted positive) in the subgroups are close to each other.
#Formula: TP / (TP + FP)
pred_rate_parity(data         = IBM, 
                 outcome      = 'Attrition',
                 group        = 'Gender',
                 probs        = 'prob', 
                 preds_levels = c('No', 'Yes'), 
                 cutoff       = 0.5, 
                 base         = 'Male')

#Accuracy parity
#Accuracy parity is achieved if the accuracies (all accurately classified divided by all predictions) in the subgroups are close to each other.
#Formula: (TP + TN) / (TP + FP + TN + FN)
acc_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

#False negative rate parity
#False negative rate parity is achieved if the false negative rates (division of false negatives with all positives) in the subgroups are close to each other.
#Formula: FN / (TP + FN)
fnr_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

#False positive rate parity
#False positive rate parity is achieved if the false positive rates (division of false positives with all negatives) in the subgroups are close to each other.
#Formula: FP / (TN + FP)
fpr_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

#Negative predictive value parity
#Negative predictive value parity is achieved if the negative predictive values (division of true negatives with all predicted negatives) in the subgroups are close to each other. This function can be considered the 'inverse' of Predictive rate parity.
#Formula: TN / (TN + FN)
npv_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')

#Specificity parity
#Specificity parity is achieved if the specificities (division of true negatives with all negatives) in the subgroups are close to each other. This function can be considered the 'inverse' of Equalized odds.
#Formula: TN / (TN + FP)
spec_parity(data         = IBM, 
            outcome      = 'Attrition',
            group        = 'Gender',
            probs        = 'prob', 
            preds_levels = c('No', 'Yes'), 
            cutoff       = 0.5, 
            base         = 'Male')
#ROC AUC comparison
#This function calculates ROC AUC and visualizes ROC curves for all subgroups. Note that probabilities must be defined for this function. Also, as ROC evaluates all possible cutoffs, the cutoff argument is excluded from this function.
roc_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           base         = 'Male')

#Matthews correlation coefficient comparison
#The Matthews correlation coefficient takes all 4 classes of the confusion matrix into consideration. According to some, it is the single most powerful metric in binary classification problems, especially for data with class imbalances.
#Formula: (TP×TN-FP×FN)/???((TP+FP)×(TP+FN)×(TN+FP)×(TN+FN))
mcc_parity(data         = IBM, 
           outcome      = 'Attrition',
           group        = 'Gender',
           probs        = 'prob', 
           preds_levels = c('No', 'Yes'), 
           cutoff       = 0.5, 
           base         = 'Male')