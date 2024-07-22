import pandas as pd
import matplotlib.pyplot as plt
from hep_ml.reweight import GBReweighter
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import numpy as np
from sklearn import preprocessing

# if we set this option to True then the input features will be standardized

# Generate some data distributed according to Gaussians
# We will have 2 sets of data with label 0 and 1 and each will have different values for the Gaussian parameters
  
# Parameters for Gaussian distributions
mean1 = 90
std_dev1 = 10
size1 = 100000

mean2 = 100
std_dev2 = 15
size2 = 120000

# Generate random data for each DataFrame
data1 = np.random.normal(mean1, std_dev1, size1)
data2 = np.random.normal(mean2, std_dev2, size2)

# Create DataFrames
df1 = pd.DataFrame({'Var': data1})
df2 = pd.DataFrame({'Var': data2})

X_target = df2['Var'].values.reshape(-1, 1)
X_original = df1['Var'].values.reshape(-1, 1)

reweighter = GBReweighter(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})

reweighter.fit(X_original,X_target)

re_weight = reweighter.predict_weights(X_original)

# renormalise after fit
normalization = len(X_target)/re_weight.sum()

re_weight*=normalization

# make a plot of distributions before and after reweighting and compare to target distribution

plt.figure(figsize=(10, 6))

lim_min = min(mean1-std_dev1*4, mean2-std_dev1*4)
lim_max = max(mean1+std_dev1*4, mean2+std_dev2*4)

plt.hist(X_original, bins=40, alpha=0.5, color='b', histtype='step', label='MC',range=(lim_min, lim_max),)
plt.hist(X_original, bins=40, alpha=0.5, color='g', histtype='step', label='MC reweighted',range=(lim_min, lim_max),weights=re_weight)
plt.hist(X_target, bins=40, alpha=0.5, color='r', histtype='step', label='Data',range=(lim_min, lim_max))
plt.legend()
plt.savefig('1D_reweighting_example.pdf')
plt.close()


