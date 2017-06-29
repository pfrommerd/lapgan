# lapgan

## Setup
Install tensorflow, keras, keras_adversarial (https://github.com/bstriner/keras-adversarial, use source, not from pip), matplotlib, pandas
The code is python3 compatible so all of the above should be install for python3

So, step by step.
### Pip
First, make sure you have pip installed (`sudo apt install python3-pip` on Ubuntu, comes installed with `brew install python3` on OSX)

### Tensorflow, Keras, Matplotlib, Pandas
Install through pip (or from source for Tensorflow for the speed boost)
```
sudo pip3 install tensorflow keras matplotlib pandas
```
### Install Keras-adversarial
```
git clone https://github.com/bstriner/keras_adversarial.git
cd keras_adversarial
python3 setup.py install
```
and you should be good to go!

## Running
To train on cifar-10, use `python3 cifar.py`, for stl10, use `python3 stl10.py`.
The tensorboard logs will be stored under output/stl or output/cifar, use `tensorboard --logdir=logs` in one of those directories to start tensorboard and connect to localhost:6006 to visualize.
For stl10, the dataset will be downloaded to data and the full, uncompressed gaussian pyramid of the data will be saved to a file so that the program can be run with considerably less memory (the data is 11gb uncompressed). Note that you need 16 GB of memory to even build uncompressed gaussian pyramid. This may change in the future so that the pyramid is built on the fly as the data is read in.
