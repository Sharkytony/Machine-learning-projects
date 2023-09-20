<pre><h1>Project : Credit Risk Assement
Author : Anthony Rodrigues </h1>

1) Data Ingestion, Importing Libraries & Data Exploration - Importing Data, Preprocessing libraries, Statistical libraries,
Visualization libraries, EDA Tool, ML MODELS, Hyperparameter Tuning libraries, Models evauluation metric libraries, 
Pipeline libraries, Exploring Statistics of the Data(mean, median, std, variance).

2)Data Cleaning with EDA - Handling Duplicates, Constant Columns Missing values under the guidance of Pandas-Profiling,
HeatMap of Correlation Matrix and Histograms showing the variance of the data and the mean, median and mode points in it. 
Exploring if the classes are not stacked together. Parallel Plot for exploring and analysing the data. One Hot Encoding 
features with cardinal values and Label encoding features with ordinal values and merging the dataframes. Using Pairplot to 
see the distribution of each feature. 

3)Statistical Processing - Differenciating features into Guassian Distribution and Skewed Distribution columns . As all 
features follow Normal distribution Applying IQR outlier detection and Percentile method for detecting outliers. Plotting 
the distribution of continous columns with lower and upper bounds. Applying Trimming and Capping on the data .

4)Data Cleaning and Preprocessing (2) - Downsampling Majority class to minority class length, Upsampling minority class to 
twice its length and merging the data. Seperating dependent and independent variables. Scaling independent variables using
StandardScaler and Applying Dimensionality Reduction .

5)Cross-validation and Train-test-split - Using Stratified KFold for cross validation, along with scikit-learn's train
  test split 

6)Model Building, Training and Evaluating - Initializing boosting, Bayes derived, KNN and Logistic regression model, Fitting
and Training models on processed data saving the results and plotting them.

7)Hyperparameter tuning - Prunning decision tree and obtaining best params Visualizing prunning (for EDA) Tuning the models
with GridSearchCV(couldnt keep in the code as took too long thus didnt include).

8)Cross Validation , Train-test-split, Model Building, Training and Evaluation (2) - Training tuned models with same Cross
Validation and Split method and plotting the evalation score.

9)Model Building, Training and Evaluating (3) - Using the Best Models in a Stacking CLassifier and Voting Classifier storing
the results and plotting them.

10)Building Pipelines - Creating python file with all functions of preprocessing (<a href="https://github.com/Sharkytony/Machine-learning-projects/blob/main/Credit%20Risk%20Assesment/preprocessor.py" target="_blank">preprocessor.py</a>).
Importing the functions Building a pipeline using them as training preprocessor(<a href="https://github.com/Sharkytony/Machine-learning-projects/blob/main/Credit%20Risk%20Assesment/train_preprocessor.pkl">train_preprocessor.pkl</a>) 
including all the steps used for processing training data, creating a python file of testing preprocessing functions 
(<a href="https://github.com/Sharkytony/Machine-learning projects/blob/main/Credit%20Risk%20Assesment/test_preprocessor.py">test_preprocessor.py</a>)with minor changes in the training functions and appending in a pipeline (<a href="https://github.com/Sharkytony/Machine-learning-projects/blob/main/Credit%20Risk%20Assesment/test_preprocessor.pkl">test_preprocessor.pkl</a>) Building Model and appending
in a pipeline<a href="https://github.com/Sharkytony/Machine-learning-projects/blob/main/Credit%20Risk%20Assesment/model.pkl">model.pkl</a> applying all the pipelines fitting, training, evaluating and storing them.


  

