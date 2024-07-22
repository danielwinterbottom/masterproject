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
#mean1 = 0
#std_dev1 = 1
#size1 = 100000
#
#mean2 = 0.2
#std_dev2 = 1.5
#size2 = 100000

mean1 = 90
std_dev1 = 10
size1 = 100000

mean2 = 100
std_dev2 = 15
size2 = 120000

# Generate random data for each DataFrame
data1 = np.random.normal(mean1, std_dev1, size1)

bkg_frac = 0.3 # fraction of background events compared to non-bkg
over_sample_bkg=10 # oversample the bkg distribution and weight down the events in the subtraction

data2_1 = np.random.normal(mean2, std_dev2, size2)
# add some background events that we will want to subtract later on
data2_2 = np.random.normal(70, 30, int(size2*bkg_frac))
bkg = np.random.normal(70, 30, int(size2*bkg_frac)*over_sample_bkg)

data2 = np.concatenate([data2_1, data2_2])

np.random.shuffle(data2)

# Create DataFrames
df1 = pd.DataFrame({'Var': data1})
df2 = pd.DataFrame({'Var': data2})
df_bkg = pd.DataFrame({'Var': bkg})

X_target = df2['Var'].values.reshape(-1, 1)
X_original = df1['Var'].values.reshape(-1, 1)
X_bkg = df_bkg['Var'].values.reshape(-1, 1)

w_target = np.ones(len(X_target))
w_bkg = -np.ones(len(X_bkg))/float(over_sample_bkg)
w_original= np.ones(len(X_original))


X_combined = np.concatenate((X_target, X_bkg), axis=0)
w_combined = np.concatenate((w_target, w_bkg), axis=0)

# Create an array of indices and shuffle it
indices = np.arange(len(X_combined))
np.random.shuffle(indices)

# Shuffle the combined data and weights using the shuffled indices
X_combined = X_combined[indices]
w_combined = w_combined[indices]

X_unmod = X_target

X_target = X_combined
w_target = w_combined

reweighter = GBReweighter(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})

reweighter.fit(X_original,X_target, w_original, w_target)

re_weight = reweighter.predict_weights(X_original)

# renormalise after fit
normalization = w_target.sum()/re_weight.sum()

re_weight*=normalization

# make a plot of distributions before and after reweighting and compare to target distribution

plt.figure(figsize=(10, 6))

lim_min = min(mean1-std_dev1*4, mean2-std_dev1*4)
lim_max = max(mean1+std_dev1*4, mean2+std_dev2*4)

plt.hist(X_original, bins=40, alpha=0.5, color='b', histtype='step', label='MC',range=(lim_min, lim_max),)
plt.hist(X_original, bins=40, alpha=0.5, color='g', histtype='step', label='MC reweighted',range=(lim_min, lim_max),weights=re_weight)
plt.hist(X_target, bins=40, alpha=0.5, color='r', histtype='step', label='Data-Bkg.',range=(lim_min, lim_max), weights=w_target)
plt.hist(X_unmod, bins=40, alpha=0.5, color='m', histtype='step', label='Data',range=(lim_min, lim_max))
plt.legend()
plt.savefig('1D_reweighting_example_negwts.pdf')
plt.close()


