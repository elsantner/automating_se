import pandas as pd
from transformers import RobertaTokenizer
from tqdm import tqdm


FLAW_LINE = "//flaw_line_below:"
FIX_LINE = "//fix_flaw_line_below:"

data_file = "./big-vul_dataset/MSR_data_cleaned.csv"

df = pd.read_csv(data_file)
df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
df.rename(columns={'vul': 'target'}, inplace=True)

all_func_before = df["func_before"].tolist()
all_func_with_fix = df["vul_func_with_fix"].tolist() 
all_del_lines = df["del_lines"].tolist()
all_target = df["target"].tolist()

# parsing the line-level data
all_processed_func = []
all_flaw_line = []
all_flaw_line_idx = []
line_len_record = []

data_len = len(all_func_with_fix)
for index in tqdm(range(data_len)):
    # skip non-vulnerable function
    # skip vulnerable function without flaw lines
    if all_target[index] == 0 or all_del_lines[index] == 0:
        all_processed_func.append(all_func_before[index])
        all_flaw_line.append("NA")
        all_flaw_line_idx.append("NA")
    else:
        # localize flaw tokens (a flaw line)
        func_with_fix = all_func_with_fix[index]
        func_with_fix = func_with_fix.split("\n")
        # localize flaw lines
        parsed_flaw_line = ""
        parsed_flaw_line_idx = ""
        # for each line in the vulnerable function
        skip_line = False
        processed_func = []
        for j in range(len(func_with_fix)):
            if skip_line:
                skip_line = False
                continue
            # if the line is a flaw line
            if func_with_fix[j].strip() == FLAW_LINE:
                # omit blank flaw line
                if func_with_fix[j+1] == "":
                    skip_line = True
                else:
                    # flaw line is the next line
                    flaw_line = func_with_fix[j+1].strip("//")
                    # append the flaw line to our processed function
                    processed_func.append(flaw_line)
                    # get the flaw line
                    parsed_flaw_line += flaw_line + "/~/"
                    # get the index of flaw line
                    parsed_flaw_line_idx += str(len(processed_func)-1) + ","
                    # skip the next line since we already processed it
                    skip_line = True
            # if the line is a fix line
            elif func_with_fix[j].strip() == FIX_LINE:
                # skip the next line
                skip_line = True
            # if the line is a clean line
            else:
                processed_func.append(func_with_fix[j].strip())
        if parsed_flaw_line == "" or \
           parsed_flaw_line_idx == "" or \
           processed_func == []:
            print("- error A when parsing line-level data")          
        # remove tailing /~/ and ,
        parsed_flaw_line = parsed_flaw_line.strip("/~/")
        parsed_flaw_line_idx = parsed_flaw_line_idx.strip(",")
        # check the flaw line data was parsed correctly
        flaw_line = parsed_flaw_line.split("/~/")
        flaw_line_idx = parsed_flaw_line_idx.split(",")
        assert len(flaw_line) == len(flaw_line_idx)
        for i in range(len(flaw_line)):
            assert flaw_line[i] == processed_func[int(flaw_line_idx[i])]
        processed_func_str = "\n".join(processed_func)
        assert processed_func_str.split("\n") == processed_func
        
        all_processed_func.append(processed_func_str)
        all_flaw_line.append(parsed_flaw_line)
        all_flaw_line_idx.append(parsed_flaw_line_idx)

# update data to df
df["processed_func"] = all_processed_func
df["flaw_line"] = all_flaw_line
df["flaw_line_index"] = all_flaw_line_idx

# write processed df to file
df.to_csv("./big-vul_dataset/processed_data.csv", index=False)

# TODO - descriptive statistics of the whole big-vul dataset
write_to_file = ""
"""
print(f"num of non-vulnerable lines: {}")
print(f"num of vulnerable lines: {}")
print(f"avg line length: {sum(line_len_record) / len(line_len_record)}")
"""
