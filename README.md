Convolutional Neural Networks with Tensorflow 
==============================================

Introduction
-------------

First and froemost, this repository has been inspired from Andrew Ng's Convolutional Neural Networks course: https://www.coursera.org/learn/convolutional-neural-networks . The python script here was written in tensorflow and was used to train on the "SIGNS" dataset 
(available in the repository). The script too has been motivated from one of Andrew Ng's assignments, but has been modified taking 
into account scalability - the pre-processing part must however be re-written if another dataset were to be used. 

In the repository, one would also find a saved model with a test accuracy of about 94%. The parametric tuning used to achieve this result was:
- Convolutional Layers : 2
- Layer 1 dimension : [4, 4, 3, 8]
- Layer 1 dropout : 0.9
- Layer 2 dimension : [2, 2, 8, 16]
- Layer 1 dropout : 1.0 
- Epochs : 500
- Learning rate : 0.001
- Batch size : 64

The script with the above tuning was run on a machine with 16 GB ram, 2GB V graphics and 1TB storage.  

Getting started
----------------
Running the script is quite straight forward and can be run directly from the terminal. The script performs three main tasks: training, comparing or testing, predicting. For each of these tasks the last part of the script must be modified - mode, path, flow etc. A few comments have been added to better articulate this part. 


So, pull the repo, teak the model and try climbing higher on the accuracy ladder. Have fun :)




#