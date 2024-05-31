import pandas as pd
import matplotlib.pyplot as plt


def plot_by_label(df, var='Z_pt',outname='Z_pt_plot.pdf'):
    # Separate the DataFrame based on the label column
    df_label_0 = df[df['label'] == 0]
    df_label_1 = df[df['label'] == 1]
   

    lim_min = df[var].quantile(0.025)
    lim_max = df[var].quantile(0.975)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(df_label_0[var], bins=50, label='MC', color='blue', weights=df_label_0['wt'],range=(lim_min, lim_max),histtype='step')
    plt.hist(df_label_1[var], bins=50, label='Data', color='red', weights=df_label_1['wt'],range=(lim_min, lim_max),histtype='step')
    
    # Add labels and title
    plt.xlabel(var)
    plt.legend()
    
    # Show the plot
    plt.savefig(outname)
    plt.close()

for i in range(10):
    file_name = 'dataframes/combined_chunk%(i)i.pkl' % vars()
    df = pd.read_pickle(file_name)
    
    print('Chuunk %(i)i' % vars())
    print(df[:10])
    
    outname = file_name.replace('.pkl','.pdf').replace('combined','Z_pt_check_combined')
    plot_by_label(df, outname=outname)
    
    outname = file_name.replace('.pkl','.pdf').replace('combined','Z_mass_check_combined')
    plot_by_label(df, outname=outname, var='Z_mass')
    
    outname = file_name.replace('.pkl','.pdf').replace('combined','U1_check_combined')
    plot_by_label(df, outname=outname, var='U1')

    outname = file_name.replace('.pkl','.pdf').replace('combined','U2_check_combined')
    plot_by_label(df, outname=outname, var='U2')
