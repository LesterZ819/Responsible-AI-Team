library(caret)
library(lime)
Attrition_test <- Attrition[1:5, 25:30]
Attrition_train <- Attrition[-(1:6), 25:30]
Attrition_lab <- Attrition[[31]][-(1:6)]
#random forest
model <- train(Attrition_train, Attrition_lab, method = 'rf')
explainer<-lime(Attrition_test,model,bin_continuous = TRUE,quantile_bins = FALSE)
explanation <- explain(Attrition_test,explainer, n_labels = 1, n_features = 4)
explanation
plot_features(explanation)
plot_explanations(explanation)
#adaboost
model <- train(Attrition_train, Attrition_lab, method = 'adaboost')
explainer<-lime(Attrition_test,model,bin_continuous = TRUE,quantile_bins = FALSE)
explanation <- explain(Attrition_test,explainer, n_labels = 1, n_features = 6)
explanation
plot_features(explanation)
plot_explanations(explanation)
#Penalized Discriminant Analysis
model <- train(Attrition_train, Attrition_lab, method = 'pda')
explainer<-lime(Attrition_test,model,bin_continuous = TRUE,quantile_bins = FALSE)
explanation <- explain(Attrition_test, explainer, n_labels = 1, n_features = 6)
explanation
plot_features(explanation)
plot_explanations(explanation)