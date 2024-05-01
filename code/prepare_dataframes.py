import uproot3
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

print(1)

def process_files_to_dataframe(file_paths, json_data, lum=59830, is_dy=False, adjust_weight=True):
    branches = ['Z_mass', 'Z_pt', 'wt']
    combined_df = pd.DataFrame()
    scaler = MinMaxScaler()
   
    for file_path in file_paths:
        json_file_name = file_path.split('/')[-1].split('_zmm_2018')[0]
        xs = evt = 1  # Default values for real data
        if adjust_weight and json_file_name in json_data: 
            xs = json_data[json_file_name]['xs']
            evt = json_data[json_file_name]['evt']

        with uproot3.open(file_path) as file:
            df = file['ntuple'].pandas.df(branches)
            if adjust_weight:  
                df['wt'] *= xs * lum / evt
                if not is_dy:
                    df['wt'] = -df['wt']
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Scale and shuffle the data before splitting into chunks
    df[['Z_mass', 'Z_pt']] = scaler.fit_transform(df[['Z_mass', 'Z_pt']])
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    return combined_df

def split_and_save_dataframe(df, output_dir, label, n_chunks=5):
    total_len = len(df)
    chunk_size = total_len // n_chunks + (total_len % n_chunks > 0)  # Calculate the size of each chunk
    
    for i in range(n_chunks):
        chunk_start = i * chunk_size
        #make sure the last chunk contains all the remaining data
        chunk_end = min((i + 1) * chunk_size, total_len)
        chunk_df = df.iloc[chunk_start:chunk_end]
        chunk_df.to_pickle(os.path.join(output_dir, f"{label}_chunk{i+1}.pkl"))



json_data = read_json('/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/params_UL_2018.json')

None_DY_list = [
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/Tbar-tW_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo1L3Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/T-tW_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W4JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W1JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKZ2Jets_ZToLL_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKWMinus2Jets_WToLNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WGToLNuG_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKWPlus2Jets_WToLNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WWTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo3LNu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo1L1Nu2Q_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WJetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/Tbar-t_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WWTo1L1Nu2Q_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W3JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo2Q2L_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W2JetsToLNu-LO_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/ZZTo4L_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/T-t_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/ZZTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTTo2L2Nu_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTToHadronic_zmm_2018.root",
    "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTToSemiLeptonic_zmm_2018.root"]


DY_list = [
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY2JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL_M-10to50-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY3JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY4JetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL-LO_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL-LO-ext1_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY1JetsToLL-LO_zmm_2018.root'
]


Real_Data_list = [
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonC_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonA_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonB_zmm_2018.root',
    '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonD_zmm_2018.root',
]

print(2)

output_dir_base = '/vols/cms/yl13923/masterproject/data_chunks'

# Process and save each category
for category, file_paths in [('None_DY', None_DY_list), ('DY', DY_list), ('Real_Data', Real_Data_list)]:
    # for Real_data，set adjust_weight to False，otherwise True
    adjust_weight = category != 'Real_Data'
    df = process_files_to_dataframe(file_paths, json_data, is_dy=(category == 'DY'), adjust_weight=adjust_weight)
    split_and_save_dataframe(df, output_dir_base, category)

# Merge mixed samples and save in chunks

# Load processed data chunks
none_dy_paths = [os.path.join(output_dir_base, f'None_DY_chunk{i+1}.pkl') for i in range(5)]
dy_paths = [os.path.join(output_dir_base, f'DY_chunk{i+1}.pkl') for i in range(5)]
real_data_paths = [os.path.join(output_dir_base, f'Real_Data_chunk{i+1}.pkl') for i in range(5)]

print(3)

# print("none_dy_paths:", none_dy_paths)
# print("dy_paths:", dy_paths)
# print("real_data_paths:", real_data_paths)

# Define a function to load and concatenate chunks
def load_and_concat_chunks(chunk_paths1, chunk_paths2):
    combined_chunks = []
    for path1, path2 in zip(chunk_paths1, chunk_paths2):
        df1 = pd.read_pickle(path1)
        df2 = pd.read_pickle(path2)
        combined_chunk = pd.concat([df1, df2], ignore_index=True)
        combined_chunks.append(combined_chunk)
    return combined_chunks

# Define a function to save chunks
def save_chunks(chunks, output_dir, label):
    for i, chunk in enumerate(chunks):
        chunk.to_pickle(os.path.join(output_dir, f"{label}_chunk{i+1}.pkl"))

# Load and merge corresponding chunks of real data and non_dy
data_chunks = load_and_concat_chunks(real_data_paths, none_dy_paths)

# Save the merged data chunks
save_chunks(data_chunks, output_dir_base, 'data')


def label_and_combine_chunks(data_chunk_path, mc_chunk_path, label_data=1, label_mc=0):
    # Load the data chunk and add a label
    data_chunk = pd.read_pickle(data_chunk_path)
    data_chunk['label'] = label_data
    
    # load the MC chunk and add a label
    mc_chunk = pd.read_pickle(mc_chunk_path)
    mc_chunk['label'] = label_mc
    
    # Merge the data and MC chunks
    combined_chunk = pd.concat([data_chunk, mc_chunk], ignore_index=True)
    
    # Randomly shuffle the merged data
    shuffled_chunk = combined_chunk.sample(frac=1).reset_index(drop=True)
    
    return shuffled_chunk

data_chunk_paths = ['/vols/cms/yl13923/masterproject/data_chunks/data_chunk1.pkl',
                     '/vols/cms/yl13923/masterproject/data_chunks/data_chunk2.pkl', 
                     '/vols/cms/yl13923/masterproject/data_chunks/data_chunk3.pkl', 
                     '/vols/cms/yl13923/masterproject/data_chunks/data_chunk4.pkl', 
                     '/vols/cms/yl13923/masterproject/data_chunks/data_chunk5.pkl']
mc_chunk_paths = ['/vols/cms/yl13923/masterproject/data_chunks/DY_chunk1.pkl', 
                  '/vols/cms/yl13923/masterproject/data_chunks/DY_chunk2.pkl', 
                  '/vols/cms/yl13923/masterproject/data_chunks/DY_chunk3.pkl', 
                  '/vols/cms/yl13923/masterproject/data_chunks/DY_chunk4.pkl', 
                  '/vols/cms/yl13923/masterproject/data_chunks/DY_chunk5.pkl']

for i, (data_path, mc_path) in enumerate(zip(data_chunk_paths, mc_chunk_paths), start=1):
    shuffled_chunk = label_and_combine_chunks(data_path, mc_path)
    
    # Save the processed data chunks
    shuffled_chunk.to_pickle(f'/vols/cms/yl13923/masterproject/data_chunks/shuffled_chunk{i}.pkl')

shuffled_chunk_paths = [
    '/vols/cms/yl13923/masterproject/data_chunks/shuffled_chunk1.pkl',
    '/vols/cms/yl13923/masterproject/data_chunks/shuffled_chunk2.pkl',
    '/vols/cms/yl13923/masterproject/data_chunks/shuffled_chunk3.pkl',
    '/vols/cms/yl13923/masterproject/data_chunks/shuffled_chunk4.pkl',
    '/vols/cms/yl13923/masterproject/data_chunks/shuffled_chunk5.pkl'
]


def split_and_save_chunks(chunk_paths, output_dir, test_size=0.95):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, chunk_path in enumerate(chunk_paths):
        data_chunk = pd.read_pickle(chunk_path)
        
        # Split the data chunk into training and validation sets
        train_chunk, val_chunk = train_test_split(data_chunk, test_size=test_size, random_state=42, stratify=data_chunk['label'])
        
        # Construct paths for saving the training and validation sets
        train_chunk_path = os.path.join(output_dir, f"train_chunk_{i+1}.pkl")
        val_chunk_path = os.path.join(output_dir, f"val_chunk_{i+1}.pkl")
        
        # save the training and validation sets
        train_chunk.to_pickle(train_chunk_path)
        val_chunk.to_pickle(val_chunk_path)

output_dir = '/vols/cms/yl13923/masterproject/data_chunks'

split_and_save_chunks(shuffled_chunk_paths, output_dir)

train_paths = [
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_1.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_2.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_3.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_4.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/train_chunk_5.pkl'
]
               
val_paths = [
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_1.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_2.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_3.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_4.pkl',
     '/vols/cms/yl13923/masterproject/data_chunks/val_chunk_5.pkl'
]