import json
import os.path

# Convert joern results from batch format to per-function format

def next_separator_index(lines, start_index):
    i = 0
    while start_index+i < len(lines):
        if lines[start_index+i] == '/*<<<SEPARATOR>>>*/\n':
            return start_index+i
        i+=1
    return -1

def get_entries(list, func_start, func_end):
    entries = []
    for e in list:
        if e >= func_start and e < func_end:
            # subtract threshold to return to single function format and -1 to convert linenum to index
            entries.append(e-func_start-1)
            
    return entries
    

def get_csv_line(func_start, func_end, json_array):
    csv_line = ""
    for list in json_array:
        linenums = get_entries(list, func_start, func_end)
        
        csv_line += '"{0}"'.format(str(linenums))
        csv_line += ','
    
    return csv_line[:-1]


if __name__ == "__main__":
    
    i=0
    csv_lines = []
    
    while os.path.isfile(r"out/batch_{0}.json".format(i)):
        
        input_file = open(r"out/batch_{0}.json".format(i))
        json_array = json.load(input_file)
        input_file.close()

        lines = []
        with open('files/batch_{0}/batch_{1}.c'.format(i,i), 'r') as file:
            lines = file.readlines()
            
        next_func_start = next_separator_index(lines,0)
        cur_func_start = 0
        
        while next_func_start != -1:
            csv_lines.append(get_csv_line(cur_func_start, next_func_start+1, json_array))
            
            cur_func_start = next_func_start+1
            next_func_start = next_separator_index(lines, cur_func_start)

        i+=1
            
            
    with open('out/syntax_features.csv', 'w') as file:
        for l in csv_lines:
            file.write(l + "\n")
    