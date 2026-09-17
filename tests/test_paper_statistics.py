import numpy as np
from sklearn.metrics import mean_absolute_error,roc_auc_score

from src.evaluation.statistics import (aggregate_case_predictions,bonferroni,
                                       delong_auc_test,paired_permutation_test,
                                       paper_statistical_test)


def test_paired_permutation_and_bonferroni_are_deterministic():
    y=np.arange(12,dtype=float);a=y+.1;b=y+np.linspace(-2,2,12)
    first=paired_permutation_test(y,a,b,mean_absolute_error,resamples=99,seed=4)
    second=paired_permutation_test(y,a,b,mean_absolute_error,resamples=99,seed=4)
    assert first==second and first[0]<0 and 0<first[1]<=1
    assert np.allclose(bonferroni([.01,.4]),[.02,.8])


def test_paired_delong_matches_auc_direction():
    y=np.array([0,0,0,0,1,1,1,1]);a=np.array([.1,.2,.3,.4,.6,.7,.8,.9]);b=np.array([.1,.8,.3,.6,.4,.7,.2,.9])
    difference,pvalue,variance=delong_auc_test(y,a,b)
    assert difference==roc_auc_score(y,a)-roc_auc_score(y,b)
    assert 0<=pvalue<=1 and variance>=0


def test_repeated_images_are_aggregated_before_paper_tests():
    case=np.array(['a','a','b','b','c','c','d','d'])
    target=np.array([0,0,0,0,1,1,1,1])
    a=np.array([.1,.3,.2,.4,.7,.9,.6,.8]);b=np.array([.4,.6,.3,.5,.5,.7,.4,.6])
    truth,case_a,case_b=aggregate_case_predictions(case,target,a,b)
    assert truth.tolist()==[0,0,1,1]
    assert np.allclose(case_a,[.2,.3,.8,.7]) and np.allclose(case_b,[.5,.4,.6,.5])
    observed,pvalue=paper_statistical_test(case,target,a,b,'binary_auc')
    assert observed==roc_auc_score(truth,case_a)-roc_auc_score(truth,case_b)
    assert 0<=pvalue<=1


def test_regression_comparison_is_mae_not_rmse():
    case=np.array(['a','a','b','b','c','c']);target=np.array([0.,0.,1.,1.,2.,2.])
    a=np.array([0.,0.,1.,1.,2.,2.]);b=np.array([1.,1.,1.,1.,1.,1.])
    observed,pvalue=paper_statistical_test(case,target,a,b,'regression_mae',resamples=31,seed=9)
    assert observed<0 and 0<pvalue<=1


def test_classification_case_labels_must_be_consistent_even_when_float_encoded():
    case=np.array(['a','a','b','b']);target=np.array([0.,1.,0.,0.]);score=np.array([.1,.2,.3,.4])
    with np.testing.assert_raises(ValueError):
        paper_statistical_test(case,target,score,score,'binary_auc')
