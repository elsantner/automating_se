import subprocess
import pandas as pd
import os

# Creates batches of function code and runs them through joern

batch_size = 25

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
      
      
if __name__ == "__main__":
    df = pd.read_csv('../big-vul_dataset/functions_only_all.csv', skipinitialspace=True, low_memory = True)

    batch = 0
    create_dir('./files/batch_{0}'.format(batch))

    func_codes = []
    i = 0
    for _, row in df.iterrows():
        func_id = row.loc['func_id']
        
        func_codes.append(row.loc['processed_func'])
        
        if i % batch_size == 0 and i != 0:
        
            with open('files/batch_{0}/batch_{0}.c'.format(batch, func_id), 'w', newline='\n') as f:
                for code in func_codes:
                    f.write(code)
                    f.write('/*<<<SEPARATOR>>>*/\n')
            
            func_codes = []        
        
            subprocess.check_call(['wsl', 'joern-parse', 'files/batch_{0}/'.format(batch)])
            subprocess.check_call(['wsl', 'joern', '--script', 'script.sc'])
            
            os.rename(r'./out/out.json', r'./out/batch_{0}.json'.format(batch))
        
            batch += 1
            create_dir('./files/batch_{0}'.format(batch))
        i+=1
   