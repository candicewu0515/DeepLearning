# NeuralNetwork & DeepLearning Model

gen_datafpeaks_2tis.ipynb
Process the narrow peak file, apply the logic of data processing and feature extraction, and generate the training/testing/validation model


Tune_hyperparams.py
Using Optuna library automated hyperparameter tuning, automatically finding the best model parameters by defining the search space and optimization


Utils.py
Supports PyTorch model training, validation, loading, and other operations. The module encapsulates commonly used tool functions


metrics.py
Calculates AUC.


neural_network.py
Describes model architecture.


Loading_data.py
Describes loaders for data.


Fitnetwork.py
Fits the network.


Test.py
A pre-trained neural network model is used to evaluate the data and finally, the performance evaluation results, such as ROC and PR curves, are generated and presented.

