# Automating Software Engineering
Trying to automate vulnerability detection on line level...

## Structure
- `process_line_level_data.py` contains a pre-processing script to add line information to the initial, cleaned dataset.
- `DataPreprocessing.ipynb` can be used to split the dataset into a _per-line_ format.
- `DataSamplingFunction.ipynb` provides label-aware sampling functionality on _function-level_.
- `DataSamplingLine.ipynb` provides label-aware sampling functionality on _line-level_.
- `MLExperimenting.ipynb` contains code for the actual ML part. (currently WIP)

## Data
This project uses the _BigVul_ dataset. The full dataset (1.5 GB compressed) can be downloaded [here](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing).
Alternatively, a already pre-processed version of the dataset on line-level (all entries and a sample) can be found [here](https://drive.google.com/drive/folders/1O6IBl6rN3U6ECROFGWGW2w0L-GztY8p5?usp=sharing).
