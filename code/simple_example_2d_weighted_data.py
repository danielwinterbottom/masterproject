import tensorflow
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import numpy as np
from sklearn import preprocessing

# if we set this option to True then the input features will be standardized
do_scaling = True

# Generate some data distributed according to Gaussians
# We will have 2 sets of data with label 0 and 1 and each will have different values for the Gaussian parameters
# Parameters for Gaussian distributions

mean1_1 = 90
std_dev1_1 = 10
mean2_1 = 0.
std_dev2_1 = 30.
rho_1 = 0.5
size_1 = 100000

mean1_2 = 100
std_dev1_2 = 15
mean2_2 = -10.
std_dev2_2 = 20.
rho_2 = 0.7
size_2 = 200000

def Generate2DGaussianData(mu_x, mu_y, sigma_x, sigma_y, rho, n_samples):
    mu = [mu_x, mu_y]
    # define coveriance matrix
    cov = [[1*sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]]

    data = np.random.multivariate_normal(mu, cov, n_samples)
    return data

data_1 = Generate2DGaussianData(mean1_1, mean2_1, std_dev1_1, std_dev2_1, rho_1, size_1)
data_2 = Generate2DGaussianData(mean1_2, mean2_2, std_dev1_2, std_dev2_2, rho_2, size_2)

df_1 = pd.DataFrame(data_1, columns=['Varx', 'Vary'])
df_2 = pd.DataFrame(data_2, columns=['Varx', 'Vary'])

# Add labels to DataFrames
df_1['Label'] = 0
df_2['Label'] = 1

# Add some weights to the dataframes to replicate what we would have in data and MC
# For data we assume most events are actual data events with weight = 1, but some (10%) are from MC subtracted events so will have negative weights around some value

# Generate 90% of weights as 1
weights_ones = np.ones(int(0.9 * size_2))
# Generate 10% of weights distributed around -2
weights_gaussian = np.random.normal(loc=-2, scale=1, size=int(0.1 * size_2))
# Concatenate the two sets of weights
weights = np.concatenate((weights_ones, weights_gaussian))
# Shuffle the weights to randomize the order
np.random.shuffle(weights)
# Add weights to the DataFrame
df_2['weight'] = weights

# for MC we produce all +ve weights around a mean value
weights_gaussian = np.random.normal(loc=1.5, scale=0.2, size=int(size_1))  
df_1['weight'] = weights_gaussian

# Concatenate DataFrames
combined_df = pd.concat([df_1, df_2])

# Display the first few rows of the combined DataFrame
print(combined_df.head())


X = combined_df[['Varx', 'Vary']]#.values#.reshape(-1, 1)
y = combined_df['Label']
w = combined_df['weight']

def Plot1DBefore(X, y, weights, var, lim_min, lim_max):
    plt.hist(X[(y == 0)][var], bins=40, alpha=0.5, color='b', label='MC',range=(lim_min, lim_max),weights=weights[(y == 0)])
    plt.hist(X[(y == 1)][var], bins=40, alpha=0.5, color='r', label='Data',range=(lim_min, lim_max),weights=weights[(y == 1)])
    plt.legend()
    plt.savefig('2D_weighted_reweighting_example_%s_before%s.pdf' % (var,'' if do_scaling else '_no_scaling'))
    plt.close()

def Plot2DHistBefore(X,y,weights,varx, vary, lim_min_x, lim_max_x, lim_min_y, lim_max_y):

  # Create subplots
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  for i in [0, 1]:
    # Select data for the current label
    Varx = X[(y == i)][varx]
    Vary = X[(y == i)][vary]

    # Create a 2D histogram for the current label
    heatmap, xedges, yedges = np.histogram2d(Varx, Vary, bins=40, range=[[lim_min_x,lim_max_x],[lim_min_y, lim_max_y]],weights=weights[(y == i)])

    # Plot the heatmap on the corresponding subplot
    ax = axes[i]
    ax.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    ax.set_title('MC' if i == 0 else 'Data')
    ax.set_xlabel('Varx')
    ax.set_ylabel('Vary')

  plt.savefig('2D_weighted_reweighting_example_%s_vs_%s_before%s.pdf' % (varx, vary,'' if do_scaling else '_no_scaling'))
  plt.close()


lim_min_x = min(mean1_1-std_dev1_1*4, mean1_2-std_dev1_2*4)
lim_max_x = max(mean1_1+std_dev1_1*4, mean1_2+std_dev1_2*4)

lim_min_y = min(mean2_1-std_dev2_1*4, mean2_2-std_dev2_2*4)
lim_max_y = max(mean2_1+std_dev2_1*4, mean2_2+std_dev2_2*4)

# make some plots of distributions 
# first plot 1D distributions

Plot1DBefore(X,y,w,'Varx',lim_min_x,lim_max_x)
Plot1DBefore(X,y,w,'Vary',lim_min_y,lim_max_y)
Plot2DHistBefore(X,y,w,'Varx', 'Vary', lim_min_x, lim_max_x, lim_min_y, lim_max_y)

if do_scaling:
  # standardize inputs 
  print('Standardizing inputs')
  scaler = preprocessing.StandardScaler().fit(X)
  scaled_data = scaler.fit_transform(X)
  X = pd.DataFrame(scaled_data, columns=X.columns, index=X.index)

# define a NN model

def simple_model(input_dimension):
    model = Sequential()
    model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization()),
    model.add(Dense(1, activation="sigmoid"))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, weighted_metrics=[])
    model.summary()
    return model

model = simple_model(2)

history = model.fit(
    X, y,
    sample_weight=w,
    validation_data=(X, y, w),
    epochs=1,#10,
    batch_size=100,
)

y_proba = model.predict(X) 

print('output scores:')
print(y_proba[:10])

re_weight = y_proba/(1.-y_proba)

print('reweights:')
re_weight = np.ravel(re_weight)
print(re_weight[:10])

if do_scaling:
  scaled_data = scaler.inverse_transform(X)
  X = pd.DataFrame(scaled_data, columns=X.columns, index=X.index)

def Plot1DAfter(X, y, weights, re_weight, var, lim_min, lim_max):
    plt.hist(X[(y == 0)][var], bins=40, alpha=0.5, color='b', label='MC',range=(lim_min, lim_max),weights=weights[(y == 0)])
    plt.hist(X[(y == 0)][var], bins=40, alpha=0.5, color='g', label='MC reweighted',range=(lim_min, lim_max),weights=re_weight[(y == 0)]*weights[(y == 0)])
    plt.hist(X[(y == 1)][var], bins=40, alpha=0.5, color='r', label='Data',range=(lim_min, lim_max),weights=weights[(y == 1)])
    plt.legend()
    plt.savefig('2D_weighted_reweighting_example_%s_after%s.pdf' % (var,'' if do_scaling else '_no_scaling'))
    plt.close()

def Plot2DHistAfter(X, y, weights, re_weights, varx, vary, lim_min_x, lim_max_x, lim_min_y, lim_max_y):
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for i in [0, 1, 2]:
        # Select data for the current label
        if i == 2:
            label = 0
        else:
            label = i
        Varx = X[(y == label)][varx].values
        Vary = X[(y == label)][vary].values

        # Extract weights

        # Create a 2D histogram for the current label
        if i == 2:
            tot_weights = re_weights[(y == label)]*weights[(y == label)]
        else: tot_weights = weights[(y == label)]
        heatmap, xedges, yedges = np.histogram2d(Varx, Vary, bins=40, 
                                                  range=[[lim_min_x, lim_max_x], [lim_min_y, lim_max_y]],
                                                  weights=tot_weights)

        # Plot the heatmap on the corresponding subplot
        ax = axes[i]
        ax.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
        if i == 0:
            title = 'MC'
        elif i == 1:
            title = 'Data'
        elif i == 2:
            title = 'MC reweighted'
        ax.set_title(title)
        ax.set_xlabel('Varx')
        ax.set_ylabel('Vary')

    plt.savefig('2D_weighted_reweighting_example_%s_vs_%s_after%s.pdf' % (varx, vary, '' if do_scaling else '_no_scaling'))
    plt.close()

Plot1DAfter(X,y,w,re_weight,'Varx',lim_min_x,lim_max_x)
Plot1DAfter(X,y,w,re_weight,'Vary',lim_min_y,lim_max_y)
Plot2DHistAfter(X,y,w,re_weight, 'Varx', 'Vary', lim_min_x, lim_max_x, lim_min_y, lim_max_y)
