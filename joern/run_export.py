import subprocess
import pandas as pd

df = pd.read_csv('../big-vul_dataset/functions_only_all.csv', skipinitialspace=True, low_memory = True)

for index, row in df.iterrows():

    with open('files/temp.c', 'w', newline='\n') as f:
        f.write(row.loc['processed_func'])
		

    subprocess.check_call(['wsl', 'joern-parse', 'files/'])
    subprocess.check_call(['wsl', 'joern', '--script', 'script.sc'])