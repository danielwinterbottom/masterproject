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

mean1 = 90
std_dev1 = 20
size1 = 100000

mean2 = 100
std_dev2 = 15
size2 = 120000

bkg_frac_1 = 0.3 # fraction of background events compared to non-bkg
over_sample_bkg_1=1 # oversample the bkg distribution and weight down the events in the subtraction

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
w_original= np.ones(len(X_original))
w_bkg_1 = -np.ones(len(X_bkg_1))/float(over_sample_bkg_1)
w_bkg_2 = -np.ones(len(X_bkg_2))/float(over_sample_bkg_2)

print('X_origional:')
print(X_original)
print('w_original:')
print(w_original)

print('X_target:')
print(X_target)
print('w_target:')
print(w_target)

def CombineAndShuffle(samples, weights):
    X_combined = np.concatenate(samples, axis=0)
    w_combined = np.concatenate(weights, axis=0)
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    w_combined = w_combined[indices]

    return X_combined, w_combined

# first we derive a weight to change origional into origional-bkg

reweighter_bkg = GBReweighter(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})

X_target_bkg, w_target_bkg = CombineAndShuffle((X_original, X_bkg_1), (w_original, w_bkg_1))

reweighter_bkg.fit(X_original, X_target_bkg, w_original, w_target_bkg)

re_weight_1 = reweighter_bkg.predict_weights(X_original)
re_weight_1*=w_original

# renormalise after fit
normalization = w_target_bkg.sum()/re_weight_1.sum()
re_weight_1*=normalization

lim_min = min(mean1-std_dev1*4, mean2-std_dev1*4)
lim_max = max(mean1+std_dev1*4, mean2+std_dev2*4)

def MakePlot(samples=[], weights=[], labels=[], outputname='output_plot.pdf', density=False):
    plt.figure(figsize=(10, 6))
    plt.hist(samples[0], bins=40, alpha=0.5, color='r', histtype='step', label=labels[0], range=(lim_min, lim_max),weights=weights[0], density=density)
    plt.hist(samples[1], bins=40, alpha=0.5, color='b', histtype='step', label=labels[1],range=(lim_min, lim_max),weights=weights[1], density=density)
    plt.hist(samples[2], bins=40, alpha=0.5, color='g', histtype='step', label=labels[2],range=(lim_min, lim_max), weights=weights[2], density=density)
    plt.legend()
    plt.savefig(outputname)
    plt.close()

MakePlot([X_original,X_target_bkg,X_original], [w_original, w_target_bkg, re_weight_1], ['Original','Origional-Bkg.','Original reweighted'], 'BDT_example_OrigToOrigMinusBkg.pdf')

# now we modify the weights for origional so that it corresponds to origional-bkg

w_original_mod = w_original*re_weight_1

# now re train a reweighter to reweight the modified origional to the target (which also includes background events) 

X_target_incsub, w_target_incsub = CombineAndShuffle((X_target, X_bkg_2), (w_target, w_bkg_2))

reweighter = GBReweighter(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})

reweighter.fit(X_original,X_target_incsub, w_original_mod, w_target_incsub)

re_weight_2 = reweighter.predict_weights(X_original)

# renormalise after fit
#normalization = w_target_incsub.sum()/re_weight_2.sum()


re_weight_2*=w_original_mod
# renormalise after fit
normalization = w_target_incsub.sum()/re_weight_2.sum()
re_weight_2*=normalization

MakePlot([X_original,X_target_incsub,X_original], [w_original_mod, w_target_incsub, re_weight_2], ['Mod. Original','Target','Mod. Original reweighted'], 'BDT_example_ModOrigToTarget.pdf', density=False)


# now we need to show that the second weight can weight origional-bkg to target-bkg
#i.e data1_1 -> data2_1

df1 = pd.DataFrame({'Var': data1_1})
df2 = pd.DataFrame({'Var': data2_1})

re_weight = reweighter.predict_weights(df1)
re_weight*=normalization


MakePlot([df1,df2,df1], [None, None, re_weight], ['Original','Target','Original reweighted'], 'BDT_example_FinalOrigToTarget.pdf', density=False)















exit()

reweighter = GBReweighter(n_estimators=100, learning_rate=0.1, max_depth=5, min_samples_leaf=1000,
                                   gb_args={'subsample': 0.4})

reweighter.fit(X_original,X_target, w_original, w_target)


#re_weight = reweighter.predict_weights(X_original)
#re_weight = 1./(1./reweighter.predict_weights(X_original) - 1./reweighter_bkg.predict_weights(X_original))

#re_weight*=w_original

# renormalise after fit
#normalization = w_target.sum()/re_weight.sum()
#re_weight*=normalization

X_original_incsub = np.concatenate((X_original, X_bkg_1), axis=0) 
w_original_incsub = np.concatenate((w_original, -w_bkg_1), axis=0) 

# Create an array of indices and shuffle it
indices = np.arange(len(X_original_incsub))
np.random.shuffle(indices)

# Shuffle the combined data and weights using the shuffled indices
X_original_incsub = X_original_incsub[indices]
w_original_incsub = w_original_incsub[indices]

print('!!!!')
print(reweighter.predict_weights(X_original_incsub))
print(reweighter_bkg.predict_weights(X_original_incsub))

#re_weight = 1./(1./reweighter.predict_weights(X_original_incsub) - 1./reweighter_bkg.predict_weights(X_original_incsub))
re_weight = 1./(1./reweighter.predict_weights(X_original_incsub) - 1./reweighter_bkg.predict_weights(X_original_incsub))
re_weight*=w_original_incsub

print ('re_weight:')
print (re_weight)

# make a plot of distributions before and after reweighting and compare to target distribution

plt.figure(figsize=(10, 6))

lim_min = min(mean1-std_dev1*4, mean2-std_dev1*4)
lim_max = max(mean1+std_dev1*4, mean2+std_dev2*4)

density=True

plt.hist(X_original, bins=40, alpha=0.5, color='b', histtype='step', label='MC-Bkg.',range=(lim_min, lim_max),weights=w_original, density=density)
plt.hist(X_original_incsub, bins=40, alpha=0.5, color='g', histtype='step', label='MC reweighted',range=(lim_min, lim_max),weights=re_weight, density=density)
plt.hist(X_target, bins=40, alpha=0.5, color='r', histtype='step', label='Data-Bkg.',range=(lim_min, lim_max), weights=w_target, density=density)
plt.hist(X_target_unmod, bins=40, alpha=0.5, color='m', histtype='step', label='Data',range=(lim_min, lim_max), density=density)
plt.hist(X_original_unmod, bins=40, alpha=0.5, color='y', histtype='step', label='MC',range=(lim_min, lim_max), density=density)
plt.legend()
plt.savefig('1D_reweighting_example_negwts_v3.pdf')
plt.close()


