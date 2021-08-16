# electricity-prediction
This project is published for the capstone project of Data-driven Artificial Intelligence based Building Energy Consumption Prediction in National University of Ireland, Galway.

The datasets in the experiments are fetched from Kaggle: https://www.kaggle.com/jeanmidev/smart-meters-in-london.

The prediction models are implemented by support vector machines (SVM), artificial neural networks (ANN), convolutional neural networks (CNN), long short-term memory (LSTM) and an ensemble model of CNN-LSTM algorithms for univariate timeseries models as well as multivariate timeseries models. Prior to the modeling, data preparation, feature selection and seasonality verification are held.

The structure of the project is as follows:
<pre>
.
|-- common_util.py
|-- plot_all_houses.py
|-- data_preparation.py
|-- feature_selection.py
|-- seasonality.py
|-- univariate_models.py
|-- multivariate_models.py
|-- smart-meters-in-london/
|   `-- ...
|-- processed/
|   |-- processedData_0.csv
|   `-- ..
|-- data/
|   |-- processedMultivariateData_0.csv
|   `-- ...
`-- plots/
    |-- original/
    |-- univariate/
    `-- multivariate/
</pre>
