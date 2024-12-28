data.py is the code for data preprocessing.
SVMmodel.py: uses the SVM model.
XBGmodel.py: uses ensemble composite models such as XGBoost, RF, and SCM.

data.py and XBGmodel.py need to be used together. data.py preprocesses the data first, and once processing is complete, XBGmodel.py is called to train the model.

SVMmodel.py can be run independently, as it integrates data preprocessing and model training.

SVM_Statics.py is the improved SVM model.
XGBmodel_noaugment.py and XGBmodelNew_noaugment.py are the improved XGB models.

The final report utilizes the model from SVM_Statics.py and, based on this, integrates auxiliary RF and FNN model.