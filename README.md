# Machine-learning-projects
Supervised + Unsupervised(Classification,Regression, Clustering, Dimensionality Reduction,Feature Engineering, EDA, Data preprocessing, Pipelines)
<pre>
Project -1(Healthcare) : Kidney Stone prediction -- Dataset link : https://www.kaggle.com/competitions/playground-series-s3e12/data
                Applied Concepts : 1)Feature Engineering(Creating new features form existing ones).
                                   2)Data cleaning and preprocessing.(Handling Missing/Null values, Handling duplicates and Outliers)
                                   3)EDA (histograms to visualize the flow of the features w.r.t. Target, scatterplot to see the ranges distincting target,
                                   boxplot to see the outliers, heatmaps to visuallize the correlation of the data)
                                   4)Train-test-split (Splitting train and validation data )/ Could have also used Stratified Kfold 
                                   5)Model Building (Scaled The data(StandardScaler- which used zscore to scale) , Applied PCA which didnt give the expected outcome,
                                   Used imp. ML models including Decision Tree, Logistic Regression, RandomForestCalssifier, XGBClassifier,SVC,KNN & GNB.
                                   6)Model Training and Checking the accuracy on the Validation Data .
                                   7)Building Pipelines for scalable models including all changes made to the Data.(Using sklearn pipeline)

Project -2(Science Fiction): Spaceship Titanic Survivors Prediction -- Dataset link : https://www.kaggle.com/competitions/spaceship-titanic/data
                Applied Concepts : 1)Feature Engineering
                                   2)Data preprocessing(Handling Null values, Handling duplicates, Encoding Features)
                                   3)EDA(Discrete columns from continous and discrete features mapping into smaller groups , Histograms, Pairplot, Countplots,Heatmaps )
                                   4)Train-test-split(Using the best split acheived by Stratified KFold for training the model for better accuracy)
                                   5)Model Building(Scaled the data using StandardScaler(), used VotingClassifier and Stacking Classifier )
                                   6)Model Training and Evaluation
                                   7)Creating Preprocessor Pipeline and Model Pipeline
                                   8)Saving and Importing the pipelines.
                                   9)Importing Pipeline and Predicting on Test data.

Project -3(Real Estate): House Price Prediction -- Dataset link : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
              Applied Concepts :  1)Feature Engineering(Created some features using logic )
                                  2)Data Preprocessing(Handling Null values, Handling duplicates, Encoding Features, Mapping Features, Multicollinearity handling,
  Feature Selection, Outlier handling since skewed data)
                                  3)EDA(Boxplots, Violinplots, Countplot, Lineplots, histograms, Heatmaps, Pieplots)
                                  4)Train-test-split(Using the best split acheived by Stratified KFold for training the model for better accuracy)
                                  5)Model Building(Scaled the data using StandardScaler(), used Gradient Boosting,RandomForest and XGBoost Regressors )
                                  6)Model Training and Evaluation( Mean Absolute Error and r2score for evaluation )
                                  7)Creating Preprocessor Pipeline and Model Pipeline
                                  8)Saving and Importing the pipelines.
                                  9)Importing Pipeline and Predicting on Test data.

Project -4(EDA):State growth insights and recommendation -- Dataset link : https://codebasics.io/challenge/codebasics-resume-project-challenge
              Applied Concepts :  1)Finding important Answers which would result in the growth 
                                  2)EDA(matplotlib, seaborn, plotly - barplots, histograms, distplots)
                                  3)Insights and Reccomendations

Project -5(Customer Segmentation): Credit card cluster segmentation and future cluster prediction 
              Applied Concepts :  1)Data cleaning and Preprocessing(handling duplicate records, missing data, constant columns, log transformation, outlier handling (percentile-based outlier detection), capping, dropping multicollinearity columns.
                                  2)EDA(histplot, heatmap, displot, kde, lineplot, scatterplot, countplot, dendogram) visualizing distribution type and other factors.
                                  3)Optimal Cluster Number, Clustering Model(Scaled using StandardScaler(), Dimensionality reduction using PCA(), Clustering by KMeans,
  and Agglomerative clustering  could also use DBSCAN but as the data is too much densed it wont work).
                                  4)Splitting the Data(Stratified Kfold, Train-Test-Split) 
                                  5)Classification Model Building(Scaled the data using StandardScaler(), using SVC and Logistric Regression)
                                  6)Training and Evaluating Model(accuracy score, confusion metrics)
                                  7)Creating Preprocessor Pipelines for Clustering and Classification and Model Pipelines for Clustering and Classification.
                                  8)Saving and Importing the pipelines.
                                  9)Importing test data and Predicting on Test data


Project -6(Fintech) :Credit Risk Assesment 
              Applied Concepts : 1) Data Ingestion and Exploration - Importing Data, Importing required libraries, About the data and statistics of the variables.
                                 2) Data Cleaning - Handling Duplicates, Constant Columns, Missing Values
                                 3) EDA - Pandas-Profiling - Barplots, Parallel plots, histograms, pairplots, correlation heatmap, displots, tree maps, lineplots, barplots
                                 4) Data Preprocessing - One Hot Encoding, Label Encoding, Outlier Handling, Handling Multicollinearity, DownSampling and UpSampling, Scaling
                                 5) Cross Validation & Train-Test-Split - Stratified KFold for acheiving best split 
                                 6) Model Training & Evaluating - Applying Scikit-learn's models (XGBoost, DecisionTree, LightGBM, GradientBoosting, RandomForest)
                                 7) Hyperparam Tuning, Model Training & Evaluating - Pruning on Decision Tree (Cutting Overfitting branches), Extracting Optimal Alpha number
for tuning Decision Tree, Tuning XGBoost, GradientBoosting, RandomForest (Boosting Algorithms).
                                 8) Pipeline - building training preprocessor pipeline, testing preprocessor pipeline, model pipeline
                                 9) Saving and Importing Pipelines  - Importing raw data, applying pipelines, fitting model, evaluating model on validation data, saving pipeline.
                                10) Training and Evaluation Pipeline - Importing Pipeline and predicting on Test data.
</pre>


                            
