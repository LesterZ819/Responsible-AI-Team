#conda create --name aif360 python=3.6
#conda activate aif360

#pip install 'aif360[all]'


#conda install -c powerai aif360
#pip install aif360==0.3.0
#pip install 'aif360[AdversarialDebiasing]'
#pip install 'aif360[LFR]'
#download: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
#          https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc
#    *place in: envs/aif360/lib/python3.6/site-packages/aif360/data/raw/german
    
    
    
    
def main():
    import sys
    sys.path.insert(1, "../")
    
    import numpy as np
    np.random.seed(0)
    
    #pip install numba==0.43.0
    #pip install --ignore-installed llvmlite==0.32.1

    from aif360.datasets import GermanDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric as CM
    from aif360.algorithms.preprocessing import Reweighing
    
    from IPython.display import Markdown, display
    
    from sklearn.ensemble import RandomForestClassifier as RF
    from sklearn.datasets import make_classification as mc 
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    
    #Step 2 Load dataset, specifying protected attribute, and split dataset into train and test
    dataset_orig = GermanDataset(protected_attribute_names=['age'],            # this dataset also contains protected attribute for "sex" 
                                                                               # which we do not consider in this evaluation
                                 privileged_classes=[lambda x: x >= 25],       # age >=25 is considered privileged
                                 features_to_drop=['personal_status','sex']   # ignore sex-related attributes
)
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    
    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]
    
    #Step 3 Compute fairness metric on original training dataset    
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,    #mean difference
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
   
    display(Markdown("#### Original training dataset"))
    print("Difference in mean outcomes between unprivileged and privileged groups = %f. AKA the privileged group is getting .17 more positive outcomes in the training dataset." % metric_orig_train.mean_difference()) #
    print()
    
    #metrics
    clf = RF()
    clf.fit(dataset_orig_train.features,dataset_orig_train.labels)
    
    predictions = clf.predict(dataset_orig_test.features)
    proba_predictions = clf.predict_proba(dataset_orig_test.features)
    
    dataset_orig_test_pred.scores = proba_predictions[:,0].reshape(-1,1)
    dataset_orig_test_pred.labels = predictions.reshape(-1, 1)
    
    cm_pred_valid = CM(dataset_orig_test, dataset_orig_test_pred, unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)
    
    cm = ["precision","recall", "accuracy"]
    
    
    metrics = {}
    for c in cm:
        metric = eval("cm_pred_valid." + c + "()")
        metrics[c] =  metric
    
    
    metrics["recall"], metrics["accuracy"], metrics["precision"]

    
    print("AIF360 metrics")
    for key in ["recall","accuracy", "precision"]:
        print("{} score is: {}".format(key,metrics[key]))
        
    #Step 4 Mitigate bias by transforming the original dataset
    RW = Reweighing(unprivileged_groups=unprivileged_groups,        #pre-processing mitigation algorithm
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)
    
   

    
    #Step 5 Compute fairness metric on transformed dataset
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
    display(Markdown("#### Transformed training dataset"))
    print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference()) #

    #metrics
    #split
    dataset_transf_train, dataset_transf_test = dataset_transf_train.split([0.7], shuffle=True)
    dataset_transf_test_pred = dataset_transf_test.copy(deepcopy=True)
    
    clf = RF()
    clf.fit(dataset_transf_train.features,dataset_transf_train.labels)
    
    predictions = clf.predict(dataset_transf_test.features)
    proba_predictions = clf.predict_proba(dataset_transf_test.features)
    
    dataset_transf_test_pred.scores = proba_predictions[:,0].reshape(-1,1)
    dataset_transf_test_pred.labels = predictions.reshape(-1, 1)
    
    cm_pred_valid = CM(dataset_transf_test, dataset_transf_test_pred, unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups)
    
    cm = ["precision","recall", "accuracy"]
    
    
    metrics = {}
    for c in cm:
        metric = eval("cm_pred_valid." + c + "()")
        metrics[c] =  metric
    
    
    metrics["recall"], metrics["accuracy"], metrics["precision"]

    
    print("AIF360 metrics")
    for key in ["recall","accuracy", "precision"]:
        print("{} score is: {}".format(key,metrics[key]))
        
main()