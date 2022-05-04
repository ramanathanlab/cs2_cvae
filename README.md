# CVAE: Protein Simulation State Embedding

## Dataset
The CVAE data is read from a folder containing two TFRecords files, `train.tfrecords` and `val.tfrecords`.
The data should be written as a serialized array of boolean values.
The folder path and the shape of each TFRecord example must be specified in [configs/params.yaml](configs/params.yaml#L8).

The data is then reshaped and cropped (upper-left corner) to be of size `input_shape`, before being fed to the model.


## Training
To train on the Cerebras System (from ANL-shared/ directory, on the host machine):
1) `cd ANL-shared/cvae/tf`
2) Train command with an orchestrator like Slurm:
    - `csrun_wse python run.py --mode train -p configs/params.yaml --model_dir $OUTPUT_DIRECTORY --cs_ip X.X.X.X`
\*note: specifying `cs_ip` instructs the run.py script to run on CS hardware.

To train on cpu/gpu (from ANL-shared/ directory):
1) `cd ANL-shared/cvae/tf`
2) `python run.py --mode train --model_dir $OUTPUT_DIRECTORY -p configs/params.yaml`


Where:  
`OUTPUT_DIR` = Path to save trained models  
* Within the ANL environment, all files required for training must be in the `/data/...` root directory path so that they will be accessible inside the container.

## Eval
To run Evaluation or prediction on any device, follow the Training instructions for that device, but pass in `eval` as the `--mode`.

## Compile-only flags:
To skip running the model and only compile the model the `--validate_only` and `--compile_only` flags can be used when running inside the cbcore container:
- `validate_only`: Compile model up to kernel matching
- `compile_only`: Compile model completely, generating compiled executables
