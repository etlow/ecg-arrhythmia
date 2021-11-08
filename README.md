# Cardiac Arrhythmia Detection
All code and environments were tested on Windows 10.

## Standalone (lite) installation
This setup runs all code in this repository.

    conda create -n arrh deepdish tqdm wfdb
    conda activate arrh
Download the required datasets.

    python data.py --download --no_save
Save and load peaks.

    python data.py --save_peaks --input_size 2048 --no_save
    python data.py --load_peaks --input_size 512
### Visualisation
Download required datasets first with the `--download` flag as above, then run this to preprocess for all input sizes.

    python visual_input.py --run_data
Visualise various input sizes.

    python visual_input.py --record 202 --sample 408000
    python visual_input.py --limit 5
## Full installation
Clone this repo and the one in references into the same base directory.

Below is the command that was used to install the environment for training with the original code.

    conda create -n emb python=3.7 keras==2.2.5 tensorflow==1.15.0 scikit-learn==0.21.3 wfdb==2.2.1 deepdish==0.3.6 scipy numpy tqdm==4.36.1 six==1.12.0 Flask==1.1.1 gevent==1.4.0 werkzeug==0.16.0 pandas=0.24.2 tensorflow-estimator=1.15.1 h5py=2.10.0 tensorflow-gpu
If that does not work, try using the exported `environment.yml`.

    conda env create -n emb -f environment.yml
Download the required datasets. To download and extract datasets into where the original implementation would, use the following commands instead.

    conda activate emb
    python data.py --physhik_path ../ecg-mit-bih/src --download --no_save
Save and load peaks.

    python data.py --physhik_path ../ecg-mit-bih/src --save_peaks --input_size 2048 --no_save
    python data.py --physhik_path ../ecg-mit-bih/src --load_peaks --input_size 512
    cd ../ecg-mit-bih/src
In `graph.py`, before the TimeDistributed layer, add either of the following lines.

    layer = MaxPooling1D(pool_size=config.input_size // 256)(layer)
    layer = Reshape((1, -1))(layer)
### Training

    python train.py --epoch 20 --input_size 512
## References
https://github.com/physhik/ecg-mit-bih
