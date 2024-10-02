import pandas as pd
import sys

def calculate_second_column_avg(file_path):
    df = pd.read_csv(file_path)
                
    avg = df.iloc[:, 1].mean()
    std = df.iloc[:, 1].std()
    return avg, std

csv_file_path = sys.argv[1]  # Replace with your CSV file path
avg,std = calculate_second_column_avg(csv_file_path)
print(avg, std)
