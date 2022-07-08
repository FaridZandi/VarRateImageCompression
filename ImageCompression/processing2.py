import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 


def get_bpp(row):
    pixels = row["width"] * row["height"]
    size = row["file_size"]
    return size / pixels * 8

fieldnames = ["name", "img_class", "setting", "height", "width", "psnr", "msssim", "bpp", "original_path", "file_path"]

all_files = ["metrics_out.csv"]    

df_from_each_file = (pd.read_csv(f, names=fieldnames) for f in all_files)
df = pd.concat(df_from_each_file, ignore_index=True)
df = df.drop_duplicates()

print(df.shape)

df = df.drop(['file_path'], axis=1)

grpbysetting = df.groupby('setting').mean()

print(grpbysetting)


grpbysetting = df.groupby('setting').mean().reset_index()

metrics = ["psnr", "msssim"]

for metric in metrics:

    baseline_codes = [
        "code_1_0", 
        "code_1_1", 
        "code_1_2", 
        "code_1_3", 
        "code_1_4", 
        "code_1_5", 
        "code_1_6", 
        "code_1_7", 
        "code_1_8", 
        "code_1_9", 
        "code_1_10", 
        "code_1_11", 
        "code_1_12", 
        "code_1_13", 
        "code_1_14", 
        "code_1_15", 
    ]

    new_codes = [
        "code_2",
        "code_3",
        "code_4",
        "code_5",
        "code_6",
        "code_7",
        "blurred",
        "original"
    ]

    baseline_rows = grpbysetting.loc[grpbysetting["setting"].isin(baseline_codes).tolist()].reset_index()
    new_rows = grpbysetting.loc[grpbysetting["setting"].isin(new_codes).tolist()].reset_index()


    ax = sns.scatterplot(data=new_rows, x="bpp", y=metric, color="red")
    ax = sns.lineplot(data=baseline_rows, x="bpp", y=metric, marker='o', color="blue")

    plt.savefig('plots/{}.png'.format(metric))


    for i in range(new_rows.shape[0]):

        if len(new_rows.setting[i]) == 6:
            annot = new_rows["setting"][i][-1]
        elif new_rows.setting[i] == "blurred":
            annot = "B"
        elif new_rows.setting[i] == "original":
            annot = "O"

        plt.text(x = new_rows.bpp[i] + 0.1,
                y = new_rows[metric][i],
                s = annot, 
            fontdict=dict(color='red',size=10),
            bbox=dict(facecolor='yellow',alpha=0.1))

    plt.savefig('plots/{}_annot.png'.format(metric))

    plt.clf()