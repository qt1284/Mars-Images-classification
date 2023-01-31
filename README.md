# Mars-Images-classification
tensorflow 2.4.0
Python 3.6.13
cs335-project

INSTALLATION

1. Log in ICL6 Machine
2. Create conda environment
    conda create -n <name> tensorflow=2.4.0 tensorflow-gpu=2.4  python=3.6
3. Install pip packages
    conda activate <name>
    python3 -m pip install --upgrade setuptools pip wheel
    python3 -m pip install nvidia-pyindex
    python3 -m pip install nvidia-cuda-runtime-cu11

4. Test installation
    python3 -c "import tensorflow as tf; print(tf.test.is_gpu_available())"
    If True then the installation of gpu was successful,
    else you can run it on cpu

5. Install the following library:
    tqdm = 4.64.1
    sklearn = 0.24.2
    numpy = 1.19.2
    matplotlib = 3.3.4
    PIL = 8.4.0

6. Download the Mars orbital image (HiRISE) labeled data set on https://zenodo.org/record/1048301

7. Change the following attribute in main() to run the model:
random_rotation=0, random_flip="none", batch_size=64, epochs=30,
            early_stopping=False, start_from_epochs = 30, confusion_matrix_plot = False
