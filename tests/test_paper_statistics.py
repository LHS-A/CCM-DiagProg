import numpy as np
from sklearn.metrics import mean_absolute_error,roc_auc_score

from src.evaluation.statistics import bonferroni,delong_auc_test,paired_permutation_test


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
