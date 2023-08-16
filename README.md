# Machine-learning-projects
Supervised + Unsupervised(Classification,Regression, Clustering,Feature Engineering, EDA, Data preprocessing)
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
                                   3)EDA(Discrete columns from continous and discrete features mapping into smaller groups , Histograms, Pairplot, Countplots,                                           Heatmaps )
                                   4)Train-test-split(Using the best split acheived by Stratified KFold for training the model for better accuracy)
                                   5)Model Building(Scaled the data using StandardScaler(), used VotingClassifier and Stacking Classifier )
                                   6)Model Training and Evaluation
                                   7)Creating Preprocessor Pipeline and Model Pipeline
                                   8)Saving and Importing the pipelines.
                                   9)Importing Pipeline and Predicting on Test data.

Project -3(Real Estate): House Price Prediction -- Dataset link : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
                                  1)Feature Engineering
                                  2)Data Preprocessing(Handling Null values, Handling duplicates, Encoding Features, Mapping Features, Multicollinearity handling)
                                  3)EDA(Boxplots, Violinplots, Countplot, Lineplots, histograps, Heatmaps )
                                  4)Train-test-split(Using the best split acheived by Stratified KFold for training the model for better accuracy)
                                  5)Model Building(Scaled the data using StandardScaler(), used Gradient Boosting,RandomForest and XGBoost Regressor's )
                                  6)Model Training and Evaluation( Mean Absolute Error and r2score for evaluation )
                                  7)Creating Preprocessor Pipeline and Model Pipeline
                                  8)Saving and Importing the pipelines.
                                  9)Importing Pipeline and Predicting on Test data.</pre>
