import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import mplhep as hep
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--n_chunks', '-n', help= 'Number of data chunks to process, -1 will run all chunks', default=2, type=int)
parser.add_argument('--input_dir', help= 'Name of input directory containing saved dataframes',default='/vols/cms/dw515/MRes_repo_2024/masterproject/dataframes_v2')
parser.add_argument('--output_dir','-o',  help= 'Name out output directory',default='model_output')
parser.add_argument('--train','-t',  help= 'If specified then the NN model will be trained, otherwise the existing model will be tested',action='store_true')
args = parser.parse_args()

n_chunks = args.n_chunks
output_dir = args.output_dir
input_dir = args.input_dir
train = args.train

if train:

    dataframes = []
    for file_path in os.listdir(input_dir):
        if file_path.endswith('.pkl') and "combined_chunk" in file_path:
            print(file_path)
            with open("%(input_dir)s/%(file_path)s" % vars(), 'rb') as file:
                df = pickle.load(file)
                dataframes.append(df)
                if n_chunks > 0 and len(dataframes) >= n_chunks: break
    df = pd.concat(dataframes, ignore_index=True)

    print('!!!!!!', len(df))

    columns = df.columns
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42,shuffle=True)
    del df

    train_df = pd.DataFrame(train_df, columns=columns)
    test_df = pd.DataFrame(test_df, columns=columns)

    # Separate features and target
    X_train = train_df.drop(columns=['label', 'wt'])
    y_train = train_df['label']
    weights_train = train_df['wt']

    X_test = test_df.drop(columns=['label', 'wt'])
    y_test = test_df['label']
    weights_test = test_df['wt']

    del train_df, test_df

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def baseline_model(input_dimension):
        model = Sequential([
            Input(shape=(input_dimension,)),
            Dense(64*2, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dense(32*2, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dense(16*2, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dense(8*2, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dense(1, activation="sigmoid")
        ])

        optimizer = Adam(learning_rate=0.0005)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, 
                    metrics=['accuracy'])
        model.summary()
        return model

    input_dim = X_train_scaled.shape[1]
    model = baseline_model(input_dim)

    history = model.fit(X_train_scaled, y_train, 
                        sample_weight=weights_train,
                        epochs=5, 
                        batch_size=1024*8, 
                        validation_data=(X_test_scaled, y_test, weights_test))

    print('Saving model...')
    model.save('%(output_dir)s/baseline_model.h5' % vars())
    print('Finished saving model...')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig('%(output_dir)s/baseline_model_accuracy.png' % vars())

    with open('%(output_dir)s/baseline_history.pkl' % vars(), 'wb') as file:
        print('Saving history...')
        pickle.dump(history.history, file)
        print('Finished saving history...')

    with open('%(output_dir)s/baseline_scaler.pkl' % vars(), 'wb') as file:
        print('Saving scaler...')
        pickle.dump(scaler, file)
        print('Finished saving scaler...')

#    with open('%(output_dir)s/baseline_test_data.pkl' % vars(), 'wb') as file:
#        print('Saving test data...')
#        pickle.dump([X_test, y_test, weights_test], file)
#        print('Finished saving test data...')
#
#    with open('%(output_dir)s/baseline_train_data.pkl' % vars(), 'wb') as file:
#        print('Saving training data...')
#        pickle.dump([X_train, y_train, weights_train], file)
#        print('Finished saving training data...')

else:

    model = tf.keras.models.load_model('%(output_dir)s/baseline_model.h5' % vars())
    with open('%(output_dir)s/baseline_test_data.pkl' % vars(), 'rb') as file:
        X_test, y_test, weights_test = pickle.load(file)

    with open('%(output_dir)s/baseline_scaler.pkl' % vars(), 'rb') as file:
        scaler = pickle.load(file)
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled, batch_size=1024*8).ravel()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, sample_weight=weights_test)

    auc = roc_auc_score(y_test, y_pred, sample_weight=weights_test)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('/vols/cms/ks1021/offline/ForDanny/masterproject/ksavva/baseline_model_roc_curve.png')

    X_test = pd.DataFrame(X_test, columns=X_test.columns)

    f = np.minimum(y_pred,0.9999)
    reweight = f/(1-f)
    X_test['weight'] = weights_test
    X_test['label'] = y_test
    X_test['factor'] = reweight

    X_test['reweight'] = X_test['weight'] * X_test['factor']

    df = X_test.copy()

    df_label_0 = df[df["label"] == 0]
    df_label_1 = df[df["label"] == 1]

    df_label_0 = df_label_0[df_label_0['Z_pt'] < 200]
    df_label_1 = df_label_1[df_label_1['Z_pt'] < 200]

    df_label_0 = df_label_0[df_label_0['Z_mass'] < 200]
    df_label_1 = df_label_1[df_label_1['Z_mass'] < 200]

    df_label_0.reset_index(drop=True, inplace=True)
    df_label_1.reset_index(drop=True, inplace=True)

    plt.style.use([hep.style.ROOT])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    hep.cms.label(rlabel="",ax=ax1)

    quantity_to_plot = "Z_mass"
    mc, bins = np.histogram(df_label_0[quantity_to_plot].to_numpy(), bins=40, weights=df_label_0["reweight"].to_numpy())
    data, bins = np.histogram(df_label_1[quantity_to_plot].to_numpy(), bins=bins, weights=df_label_1["weight"].to_numpy())
    mc_original, bins = np.histogram(df_label_0[quantity_to_plot].to_numpy(), bins=bins, weights=df_label_0["weight"].to_numpy())

    hep.histplot(
        data, # need to calculate errors properly
        bins=bins,
        histtype="errorbar", 
        color="black",
        label="Data",
        ax=ax1,
    )

    hep.histplot(
        mc,
        bins=bins,
        histtype="step",
        color="blue",
        label="MC_Reweighted",
        ax=ax1,
    )

    hep.histplot(
        mc_original,
        bins=bins,
        histtype="step",
        color="red",
        label="MC",
        ax=ax1,
    )

    bin_counts_mc, _ = np.histogram(df_label_0[quantity_to_plot], bins=bins, weights=df_label_0["reweight"])
    bin_counts_mc_original, _ = np.histogram(df_label_0[quantity_to_plot], bins=bins, weights=df_label_0["weight"])
    bin_counts_data, _ = np.histogram(df_label_1[quantity_to_plot], bins=bins, weights=df_label_1["weight"])
    bin_centers = (bins[1:] + bins[:-1]) / 2

    ratio = bin_counts_mc / (bin_counts_data + 1e-6)
    ratio_original = bin_counts_mc_original / (bin_counts_data + 1e-6)

    hep.histplot(
        ratio,
        bins=bins,
        yerr=0, # need to calculate errors properly
        histtype="errorbar",
        color="blue",
        label="MC_Reweighted/Data",
        ax=ax2,
    )

    hep.histplot(
        ratio_original,
        yerr=0, # need to calculate errors properly
        bins=bins,
        histtype="errorbar",
        color="red",
        label="MC/Data",
        ax=ax2,
    )

    ax1.set_ylabel("Events\n", fontsize=25)
    ax1.legend(fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20) 

    xlabels = {
        "Z_pt": r"$Z_{p_{T}}$ (GeV)",
        "Z_mass": r"$Z_{mass}$ (GeV)"
    }
    ax2.set_ylabel("Ratio", fontsize=25)
    ax2.set_xlabel(xlabels[quantity_to_plot], fontsize=25)
    ax2.axhline(1, color='gray', linestyle='--')
    ax2.set_ylim(0.5, 1.5)

    fig.savefig(f"/vols/cms/ks1021/offline/ForDanny/masterproject/ksavva/plots/{quantity_to_plot}.png",dpi=140)
