# jayde16_olvea16_Bachelor2019
Repository for program files for the project 'Prediction of the DAS28 score based on EULAR-OMERACT scores using neural networks' written spring 2019 by Jakob Yde-Madsen and Oliver Vea.

Folder structure:

## Matlab

- ### AdditionalDataRegression
  - ##### newdata_augmented.m
    This file is used to create the augmented data set, do linear multivariate regression using mvregress and lsqnonneg and test the regression on the test set.
  - ##### newdata_full.m
    This file is used to filter the data set, such that only the full data points remain, do linear multivariate regression using mvregress and lsqnonneg and test the regression on the test set.

- ### AverageBaseline
  - ##### DAS28random.m
    This predicts the DAS28 scores in both test sets using the average from the training set, it also prints the MAE and MSE.

- ### NQQPlot
  - ##### drawHistograms.m
    This file draws qqplots and a scatter plot comparing the data in filtered.csv to a normal distribution.

- ### Regression
  - ##### Regression_augmented.m
    This file is used to create the augmented data set, do linear multivariate regression using mvregress and lsqnonneg and test the regression on the test set.
  - ##### Regression_full.m
    This file is used to filter the data set, such that only the full data points remain, do linear multivariate regression using mvregress and lsqnonneg and test the regression on the test set.

- ### RegressionDistTest
  - ##### testtrainvarmean.m
    This file is used to analyze the DAS28 and CRP values of the dataset.


## Python
- ### CSVReader
  - ##### CSVReader.py
  - ##### inc_exampledata.csv

- ### FileConversion
  - ##### npy_to_csv.py
    This file convertes the data format from .npy to .csv.

- ### NNTuning
  - #### BroadSearch
    - ##### NNBroadSearch.py
      This file is used for the first NN search.
      
  - #### CreateFinalModels
    - ##### CreateFinalModels.py
      This file is used to create the final models used to predict the DAS28 scores.   
      
  - #### FinalTest
    - ##### FinalTest.py
      This file is used to test the final model using different initializations.
    - ##### ModelTest.py
      This file is used to predict the models' performance on the first test set.
    - ##### TestModelNewData.py
      This file is used to predict the models' performance on the second test set.
      
  - #### FineTuningTop10
    - ##### NNfinetuneTop10.py
      This file is used for the second NN test.
    - ##### inc_bestmodels.csv
      Best models from the first test, used to run the fine-tuning test.
  - #### LearningRateTest
    - ##### Learning_rate.py
      This is used for the learning_rate test.
      
  - #### WeightInitTest
    - ##### Different_seeds.py
      This is used to test the same model using different seeds.
    - ##### sameseed.py
      This is used to test the same model using the same seed.

- ### PlotLy
  - ##### PlotLy.py
    This file allows for plotting parallel coordinates, scatterplots and 2d histogram plots.
  - ##### inc_filtered.csv
    An example of an input file for the plotting program.

- ### RNNTuning
  - ##### RNNTuning.py
    This model is used as the initial tuning of the LSTM-based model.
  - ##### inc_bestmodels.csv
    This file contains the best model for the rough search.
    
- ### LSTM Tuning
  - ##### LSTM_Tuning.py
    This file is used for the final tuning of the LSTM-based model.
    
- ### DataToDict
  - ##### npy_to_csv.py
    This file convertes the data format from .csv to .npy, storing data in a dict in between.
    
- ### ModelTest
  - ##### ModelTest.py
    This script tests the performance of the models in the models folder on the old test set.
  - ##### models
    This folder contains the models to be tested.
   
- ### TestModelNewData
  - ##### TestModelNewData.py
    This script tests the performance of the models in the models folder on the new test set.
  - ##### models
    This folder contains the models to be tested.
    
- ### DataMapping
  - ##### DataMapping.py
    Script mapping the EULAR-OMERACT scores to the CRP and DAS28 scores.
  - ##### Session.py
    File describing the Session class storing information about each scanning session.
  - ##### ImageInformation.py
    File describing the imageinformation class storing information about the EULAR-OMERACT scores.
    

