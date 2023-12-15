# hu_prato_mutlivariate_psd_estimation


Summary：
    - The "code" file includes codes and data for "Fast Bayesian Inference on Spectral Analysisof Multivariate Stationary Time Series" by Zhixiong Hu and Raquel Prado.
    - The codes are written in Python 3.7.0 (suggested version >= 3.0).    
    - Author: Zhixiong Hu


Structure of the files:
    - "simulation_study_demo.py" and "data_analysis_demo.py" are runbooks to show how to use our approach to reproduce part of the results in the paper.
    - "model" file has the model module "spec_vi.py" that defines the framework and model.
    - "data" file includes example data and its processing scripts.

Quick guide:
To run the runbooks, two things needs to be done:
    1) Set "code" as the work directory.
    2) Install relevant packages:
        - pip install --upgrade tensorflow (suggested version >= 2.1) 
	  (The code runs on either CPU or GPU. But GPU usage requires extra GPU setup, which is beyond the workload of this demo.)
	- pip install --upgrade tensorflow-probability (suggested version >= 0.9)
	- pip install pandas (suggested version >= 1.0.3)
	- pip install numpy (suggested version >= 1.19.5)
	- (pip install scipy if necessary, this package should be included in python by default)

To download and use python, we suggest using Anaconda and the Spyder IDE inside Anaconda to run the .py files. 
Useful linkes:
    - Anaconda: https://www.anaconda.com/products/individual   
    - Tensorflow: https://www.tensorflow.org/api_docs (Tensorflow GPU setup: https://www.tensorflow.org/install/gpu)
    - tensorflow-probability: https://www.tensorflow.org/probability

**Note**
    We are actively updating the code. For example, currently the name convention in "spec_vi.py" is messy. In the next step, we want to follow PeP-8 style to improve the readability and consistency of our Python code.