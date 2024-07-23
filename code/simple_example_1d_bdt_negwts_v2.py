import pandas as pd
import matplotlib.pyplot as plt
from hep_ml.reweight import GBReweighter
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import numpy as np
from sklearn import preprocessing

# in this example we add negative weights for both target and origional

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

bkg_frac_1 = 0.3 # fraction of background events compared to non-bkg
over_sample_bkg_1=3 # oversample the bkg distribution and weight down the events in the subtraction

# Generate random data for each DataFrame
data1_1 = np.random.normal(mean1, std_dev1, size1)
data1_2 = np.random.normal(100, 30, int(size1*bkg_frac_1))
bkg_1 = np.random.normal(100, 30, int(size1*bkg_frac_1)*over_sample_bkg_1)

data1 = np.concatenate([data1_1, data1_2])
np.random.shuffle(data1)

bkg_frac_2 = 0.3 # fraction of background events compared to non-bkg
over_sample_bkg_2=3 # oversample the bkg distribution and weight down the events in the subtraction

data2_1 = np.random.normal(mean2, std_dev2, size2)
# add some background events that we will want to subtract later on
data2_2 = np.random.normal(70, 30, int(size2*bkg_frac_2))
bkg_2 = np.random.normal(70, 30, int(size2*bkg_frac_2)*over_sample_bkg_2)

data2 = np.concatenate([data2_1, data2_2])

np.random.shuffle(data2)

# Create DataFrames
df1 = pd.DataFrame({'Var': data1})
df2 = pd.DataFrame({'Var': data2})
df_bkg_1 = pd.DataFrame({'Var': bkg_1})
df_bkg_2 = pd.DataFrame({'Var': bkg_2})

X_target = df2['Var'].values.reshape(-1, 1)
X_original = df1['Var'].values.reshape(-1, 1)
X_bkg_1 = df_bkg_1['Var'].values.reshape(-1, 1)
X_bkg_2 = df_bkg_2['Var'].values.reshape(-1, 1)

w_target = np.ones(len(X_target))
w_bkg_1 = -np.ones(len(X_bkg_1))/float(over_sample_bkg_1) # put back to negative!!!
w_bkg_2 = -np.ones(len(X_bkg_2))/float(over_sample_bkg_2)
w_original= np.ones(len(X_original))


X_target_combined = np.concatenate((X_target, X_bkg_2), axis=0)
w_target_combined = np.concatenate((w_target, w_bkg_2), axis=0)

# Create an array of indices and shuffle it
indices = np.arange(len(X_target_combined))
np.random.shuffle(indices)

# Shuffle the combined data and weights using the shuffled indices
X_target_combined = X_target_combined[indices]
w_target_combined = w_target_combined[indices]

X_target_unmod = X_target

X_target = X_target_combined
w_target = w_target_combined

X_original_combined = np.concatenate((X_original, X_bkg_1), axis=0)
w_original_combined = np.concatenate((w_original, w_bkg_1), axis=0)

# Create an array of indices and shuffle it
indices = np.arange(len(X_original_combined))
np.random.shuffle(indices)

# Shuffle the combined data and weights using the shuffled indices
X_original_combined = X_original_combined[indices]
w_original_combined = w_original_combined[indices]

X_original_unmod = X_original

X_original = X_original_combined
w_original = w_original_combined

print('X_origional:')
print(X_original)
print('w_original:')
print(w_original)

print('X_origional:')
print(X_target)
print('w_target:')
print(w_target)

reweighter = GBReweighter(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})

reweighter.fit(X_original,X_target, w_original, w_target)

re_weight = reweighter.predict_weights(X_original)

re_weight*=w_original

# renormalise after fit
normalization = w_target.sum()/re_weight.sum()
re_weight*=normalization


#re_weight[w_original < 0] = w_original[w_original < 0]

print ('re_weight:')
print (re_weight)

# make a plot of distributions before and after reweighting and compare to target distribution

plt.figure(figsize=(10, 6))

lim_min = min(mean1-std_dev1*4, mean2-std_dev1*4)
lim_max = max(mean1+std_dev1*4, mean2+std_dev2*4)

density=False

plt.hist(X_original, bins=40, alpha=0.5, color='b', histtype='step', label='MC-Bkg.',range=(lim_min, lim_max),weights=w_original, density=density)
plt.hist(X_original, bins=40, alpha=0.5, color='g', histtype='step', label='MC reweighted',range=(lim_min, lim_max),weights=re_weight, density=density)
plt.hist(X_target, bins=40, alpha=0.5, color='r', histtype='step', label='Data-Bkg.',range=(lim_min, lim_max), weights=w_target, density=density)
plt.hist(X_target_unmod, bins=40, alpha=0.5, color='m', histtype='step', label='Data',range=(lim_min, lim_max), density=density)
plt.hist(X_original_unmod, bins=40, alpha=0.5, color='y', histtype='step', label='MC',range=(lim_min, lim_max), density=density)
plt.legend()
plt.savefig('1D_reweighting_example_negwts_v2.pdf')
plt.close()


