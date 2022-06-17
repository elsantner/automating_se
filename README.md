# Automating Software Engineering
Trying to automate vulnerability detection on line level...

## Structure
- `Results/`: performance results obtained throughout development
- `joern/`: artifacts of experimenting with Joern
- `models/`: some pretrained models (might be outdated)
- `vocab/`: pre-extracted vocabulary files for CountVectorizer
- `DataPreprocessing.ipynb`: can be used to split the dataset into a _per-line_ format
- `DataSamplingFunction.ipynb`: label-aware sampling functionality on _function-level_
- `DataSamplingLine.ipynb`: label-aware sampling functionality on _line-level_
- `Function_Vul_Prediction.ipynb`: function-level performance evaluation
- `MLExperimenting_BERT.ipynb`: machine learning implementation of our approach
- `MLFunctionModel.ipynb`: training of function model
- `function_encoders.py`: different encoders for whole functions
- `line_encoders.py`: different encoders for lines
- `process_line_level_data.py`: contains a pre-processing script to add line information to the initial, cleaned dataset.

## Data
This project uses the _BigVul_ dataset. The full dataset (1.5 GB compressed) can be downloaded [here](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing).
Alternatively, a already pre-processed version of the dataset on line-level (all entries and a sample) can be found [here](https://drive.google.com/drive/folders/1O6IBl6rN3U6ECROFGWGW2w0L-GztY8p5?usp=sharing).
