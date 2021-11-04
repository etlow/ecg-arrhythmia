# Cardiac Arrhythmia Detection
Instructions on running will be added soon
All code and packages tested on Windows 10

## Standalone (lite) installation
Can run all code in this repository

    conda create -n arrh deepdish tqdm wfdb
    conda activate arrh
Download required datasets

    python data.py --download --no_save
Save and load peaks

    python data.py --save_peaks --input_size 2048 --no_save
    python data.py --load_peaks --input_size 512
### Visualisation
Download required datasets first, then run this to preprocess for all input sizes

    python visual_input.py --run_data
To show graphs

    python visual_input.py --record 202 --sample 408000
    python visual_input.py --limit 5
## Full installation
`environment.yml` should work for training purposes in the original implementation

## References
https://github.com/physhik/ecg-mit-bih
