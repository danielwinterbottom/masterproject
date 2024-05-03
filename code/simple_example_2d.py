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
size_2 = 100000

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

# Concatenate DataFrames
combined_df = pd.concat([df_1, df_2])

# Display the first few rows of the combined DataFrame
print(combined_df.head())


X = combined_df[['Varx', 'Vary']]#.values#.reshape(-1, 1)
y = combined_df['Label']


def Plot1DBefore(X, y, var, lim_min, lim_max):
    plt.hist(X[(y == 0)][var], bins=40, alpha=0.5, color='b', label='MC',range=(lim_min, lim_max),)
    plt.hist(X[(y == 1)][var], bins=40, alpha=0.5, color='r', label='Data',range=(lim_min, lim_max))
    plt.legend()
    plt.savefig('2D_reweighting_example_%s_before%s.pdf' % (var,'' if do_scaling else '_no_scaling'))
    plt.close()

def Plot2DHistBefore(X,y,varx, vary, lim_min_x, lim_max_x, lim_min_y, lim_max_y):

  # Create subplots
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  for i in [0, 1]:
    # Select data for the current label
    Varx = X[(y == i)][varx]
    Vary = X[(y == i)][vary]

    # Create a 2D histogram for the current label
    heatmap, xedges, yedges = np.histogram2d(Varx, Vary, bins=40, range=[[lim_min_x,lim_max_x],[lim_min_y, lim_max_y]])

    # Plot the heatmap on the corresponding subplot
    ax = axes[i]
    ax.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    ax.set_title('MC' if i == 0 else 'Data')
    ax.set_xlabel('Varx')
    ax.set_ylabel('Vary')

  plt.savefig('2D_reweighting_example_%s_vs_%s_before%s.pdf' % (varx, vary,'' if do_scaling else '_no_scaling'))
  plt.close()


lim_min_x = min(mean1_1-std_dev1_1*4, mean1_2-std_dev1_2*4)
lim_max_x = max(mean1_1+std_dev1_1*4, mean1_2+std_dev1_2*4)

lim_min_y = min(mean2_1-std_dev2_1*4, mean2_2-std_dev2_2*4)
lim_max_y = max(mean2_1+std_dev2_1*4, mean2_2+std_dev2_2*4)

# make some plots of distributions 
# first plot 1D distributions

Plot1DBefore(X,y,'Varx',lim_min_x,lim_max_x)
Plot1DBefore(X,y,'Vary',lim_min_y,lim_max_y)
Plot2DHistBefore(X,y,'Varx', 'Vary', lim_min_x, lim_max_x, lim_min_y, lim_max_y)

print(X[(y == 0)])
if do_scaling:
  # standardize inputs 
  print('Standardizing inputs')
  scaler = preprocessing.StandardScaler().fit(X)
  scaled_data = scaler.fit_transform(X)
  X = pd.DataFrame(scaled_data, columns=X.columns, index=X.index)

# define a NN model


print(X[(y == 0)])

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
    validation_data=(X, y),
    epochs=10,
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

def Plot1DAfter(X, y, re_weight, var, lim_min, lim_max):
    plt.hist(X[(y == 0)][var], bins=40, alpha=0.5, color='b', label='MC',range=(lim_min, lim_max),)
    plt.hist(X[(y == 0)][var], bins=40, alpha=0.5, color='g', label='MC reweighted',range=(lim_min, lim_max),weights=re_weight[(y == 0)])
    plt.hist(X[(y == 1)][var], bins=40, alpha=0.5, color='r', label='Data',range=(lim_min, lim_max))
    plt.legend()
    plt.savefig('2D_reweighting_example_%s_after%s.pdf' % (var,'' if do_scaling else '_no_scaling'))
    plt.close()

def Plot2DHistAfter(X, y, re_weights, varx, vary, lim_min_x, lim_max_x, lim_min_y, lim_max_y):
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
        weights = re_weights[(y == label)]

        # Create a 2D histogram for the current label
        if i == 2:
            heatmap, xedges, yedges = np.histogram2d(Varx, Vary, bins=40, 
                                                      range=[[lim_min_x, lim_max_x], [lim_min_y, lim_max_y]],
                                                      weights=weights)
        else:
            heatmap, xedges, yedges = np.histogram2d(Varx, Vary, bins=40, 
                                                      range=[[lim_min_x, lim_max_x], [lim_min_y, lim_max_y]])

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

    plt.savefig('2D_reweighting_example_%s_vs_%s_after%s.pdf' % (varx, vary, '' if do_scaling else '_no_scaling'))
    plt.close()

Plot1DAfter(X,y,re_weight,'Varx',lim_min_x,lim_max_x)
Plot1DAfter(X,y,re_weight,'Vary',lim_min_y,lim_max_y)
Plot2DHistAfter(X,y,re_weight, 'Varx', 'Vary', lim_min_x, lim_max_x, lim_min_y, lim_max_y)
