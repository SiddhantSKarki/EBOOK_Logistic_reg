import pandas as pd
import  argparse
import os.path


p = argparse.ArgumentParser()
p.add_argument('--CSV', '-c', help="Te CSV file name (without .csv) default is export.csv", default="export")
p.add_argument('--export_CSV', '-e', help="Exporting CSV file name", default="filtered_data")
args = p.parse_args()

csv_file_name = args.CSV

df = pd.read_csv(f"{csv_file_name}.csv",comment='#',  header=None, skipinitialspace=True)

def name_status_extraction(df):
    df.columns = df.iloc[0]
    df.drop(0, axis=0, inplace=True)
    new_df = pd.DataFrame([df["BookTitle"], df["Work Order Status"]]).T
    new_df.columns = ["book_name", "status"]
    return new_df

filtered_df = name_status_extraction(df)

status = filtered_df.iloc[2][1]

#Creating a boolean mask
filtered_df["is_ebook"] = ((filtered_df["status"]) == status)

def export_file(df):
    exp_file = args.export_CSV
    file_num = 1
    if (os.path.exists("filtered_data.csv")):
        print("File already exists! Choose a different name")
    else:
        df.to_csv(f"{exp_file}.csv", index=False)
        print(f"{exp_file}.csv successfully created")

export_file(filtered_df)