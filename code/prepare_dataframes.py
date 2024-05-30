import uproot
import pandas as pd
import json
import os

variables = ['Z_mass', 'Z_pt', 'wt', 'n_jets', 'n_deepbjets', 'mjj', 'jdeta', 'jdphi', 'dijetpt', 'jpt_1', 'jpt_2', 'jpt_3'] 
json_file = '/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/params_UL_2018.json'
output_dir='dataframes'
nchunks=10
verbosity = 2

def read_json(file_path):
    if verbosity > 0: print('reading in json from %(file_path)s' % vars())
    with open(file_path) as f:
        data = json.load(f)
    return data

def read_root_in_chunks(filename, json_file, output_dir='dataframes', nchunks=10, isMC=False, isSubtracted=False):
    sample_name = f.split('/')[-1].replace('_zmm_2018.root','')
    # read in json for weighting mc samples
    json_data = read_json(json_file)
    lum = json_data['SingleMuon']['lumi']

    if verbosity > 0: print('Preparing dataframes for sample: %s using %i chunks' %(f,nchunks))
    # Open the root file
    tree = uproot.open(filename)["ntuple"]

    # Get the total number of entries
    num_entries = tree.num_entries

    chunksize = int(num_entries/nchunks)

    if verbosity > 0: print('Total events in sample = %(num_entries)i, number of events per chunk = %(chunksize)g' % vars())

    # Iterate over the chunks
    for i in range(nchunks):
        # Calculate start and stop indices for the current chunk
        start = i * chunksize
        stop = min((i + 1) * chunksize, num_entries)
        if verbosity > 1: print ('Processing chunk %(i)i, start and stop indices = %(start)s and %(stop)s' % vars())

        # Read the current chunk into a dataframe
        df = tree.arrays(variables, library="pd", entry_start=start, entry_stop=stop) 

        if isMC:
            # for MC events adjust weights so that we scale everything to the cross-section * luminosity
            xs = json_data[sample_name]['xs']
            evt = json_data[sample_name]['evt']
            df['wt'] *= xs * lum / evt
            if isSubtracted:
                # if sample is to be a subtracted component e.g subracting some MC background from the data then we flip the sign of the weights
                df['wt'] = -df['wt']

        # now we add labels, data has label =1, MC has label=0
        # for the subtracted MC events these will also get label=1 as for the data, since they are to be used to subtract background contributions from the data
        if isMC and not isSubtracted: 
            df['label'] = 0
        else:    
            df['label'] = 1

        if verbosity > 1: 
            print('Number of events in chunk = %i' % len(df))

            print("First Entry in chunk:")
            print(df.head(1))

            print("Last Entry in chunk:")
            print(df.tail(1))

        # save dataframe then delete

        output_name = '%(output_dir)s/%(sample_name)s_chunk%(i)i.pkl' % vars()

        if verbosity > 1: print('Writing %(output_name)s\n' % vars())
        
        df.to_pickle(output_name)
        del df

def find_files_ending_with(directory, suffix):
    """Find all files in the given directory that end with the specified suffix."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)]

def load_and_concatenate_dataframes(file_paths):
    """Load and concatenate pandas DataFrames from a list of file paths."""
    dataframes = []
    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            df = pd.read_pickle(file)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


background_samples = [
                     "/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/Tbar-tW_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo1L3Nu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/T-tW_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W4JetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W1JetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKZ2Jets_ZToLL_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKWMinus2Jets_WToLNu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WGToLNuG_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/EWKWPlus2Jets_WToLNu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WWTo2L2Nu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo3LNu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo1L1Nu2Q_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WJetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/Tbar-t_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WWTo1L1Nu2Q_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W3JetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/WZTo2Q2L_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/W2JetsToLNu-LO_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/ZZTo4L_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/T-t_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/ZZTo2L2Nu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTTo2L2Nu_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTToHadronic_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/TTToSemiLeptonic_zmm_2018.root"
                     ]
MC_samples=[
"/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY2JetsToLL-LO_zmm_2018.root",
"/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL_M-10to50-LO_zmm_2018.root",
"/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY3JetsToLL-LO_zmm_2018.root",
"/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY4JetsToLL-LO_zmm_2018.root",
"/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL-LO_zmm_2018.root",
"/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DYJetsToLL-LO-ext1_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/DY1JetsToLL-LO_zmm_2018.root"]

data_samples=["/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonC_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonA_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonB_zmm_2018.root","/vols/cms/dw515/outputs/MRes/MRes_2024_Run2018_v3/SingleMuonD_zmm_2018.root"]

# first process and save each sample as a dataframe split into chunks

# Process and save each category
for category, file_paths in [('background', background_samples), ('MC', MC_samples), ('Real_Data', data_samples)]:
    isMC = category != 'Real_Data'
    isSubtracted = category == 'background'
    for f in file_paths: 
        read_root_in_chunks(f,nchunks=nchunks,output_dir=output_dir,json_file=json_file,isMC=isMC,isSubtracted=isSubtracted)

#Once all data is read we will combine each set of chunks into a single combined dataframe 

for i in range(nchunks):
    if verbosity>0: print('combining data for chunk %(i)i' % vars())
    suffix = 'chunk%(i)i.pkl' % vars()
    output_file = '%(output_dir)s/combined_chunk%(i)i.pkl' % vars()

    file_paths = find_files_ending_with(output_dir, suffix)

    if verbosity>1:
        print('The following files will be combined:')
        for f in file_paths: print(f)

    # Load and concatenate all the DataFrames
    combined_df = load_and_concatenate_dataframes(file_paths)
    # Shuffle the combined DataFrame
    combined_df = combined_df.sample().reset_index(drop=True)

    # Save the shuffled DataFrame to a pickle file
    combined_df.to_pickle(output_file)
    del combined_df
    
