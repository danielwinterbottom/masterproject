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
mean1 = 0
std_dev1 = 1
size1 = 100000

mean2 = 0.2
std_dev2 = 1.5
size2 = 100000

mean1 = 90
std_dev1 = 10
size1 = 100000

mean2 = 100
std_dev2 = 15
size2 = 100000

# Generate random data for each DataFrame
data1 = np.random.normal(mean1, std_dev1, size1)
data2 = np.random.normal(mean2, std_dev2, size2)

# Create DataFrames
df1 = pd.DataFrame({'Var': data1})
df2 = pd.DataFrame({'Var': data2})

# Add labels to DataFrames
df1['Label'] = 0
df2['Label'] = 1

# Concatenate DataFrames
combined_df = pd.concat([df1, df2])

# Display the first few rows of the combined DataFrame
print(combined_df.head())

X = combined_df['Var'].values.reshape(-1, 1)
y = combined_df['Label']

if do_scaling:
  # standardize inputs 
  print('Standardizing inputs')
  scaler = preprocessing.StandardScaler().fit(X)
  X = scaler.transform(X)

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

model = simple_model(1)

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


# make a plot of distributions before and after reweighting and compare to target distribution

plt.figure(figsize=(10, 6))

lim_min = min(mean1-std_dev1*4, mean2-std_dev1*4)
lim_max = max(mean1+std_dev1*4, mean2+std_dev2*4)

if do_scaling:
  X = scaler.inverse_transform(X)

plt.hist(X[(y == 0)], bins=40, alpha=0.5, color='b', label='MC',range=(lim_min, lim_max),)
plt.hist(X[(y == 0)], bins=40, alpha=0.5, color='g', label='MC reweighted',range=(lim_min, lim_max),weights=re_weight[(y == 0)])
plt.hist(X[(y == 1)], bins=40, alpha=0.5, color='r', label='Data',range=(lim_min, lim_max))
plt.legend()
plt.savefig('1D_reweighting_example%s.pdf' % ('' if do_scaling else '_no_scaling'))
plt.close()

