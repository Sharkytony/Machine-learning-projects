<pre><b><h1>
Project : Credit Risk Assessment
Author: Anthony Rodrigues</h1>
<h2>
Section 1: Data Preparation and Exploration
</h2>

<h3>
1.1 Data Ingestion and Library Setup
</h3>
Importing Data
Preprocessing Libraries
Statistical Libraries
Visualization Libraries
<h3>
1.2 Exploratory Data Analysis (EDA)
</h3>
EDA Tool
Statistical Data Overview
Mean, Median, Standard Deviation, Variance
Data Visualization
Histograms
Correlation Matrix Heatmap
Parallel Plot
Pairplot
<h2>
Section 2: Data Cleaning and Preprocessing
</h2>
<h3>
2.1 Handling Data Quality Issues
</h3>
Duplicate Handling
Handling Constant Columns
Handling Missing Values (Pandas-Profiling)
<h3>
2.2 Data Transformation
</h3>
One-Hot Encoding for Cardinal Features
Label Encoding for Ordinal Features
Merging Dataframes
<h3>
2.3 Outlier Detection and Treatment
</h3>
Identifying Gaussian and Skewed Distributions
Outlier Detection using IQR and Percentile Methods
Trimming and Capping Outliers
<h3>
2.4 Data Balancing
</h3>
Downsampling Majority Class
Upsampling Minority Class
Merging Balanced Data
Feature Scaling using StandardScaler
Dimensionality Reduction
<h2>
Section 3: Model Development
</h2>
<h3>
3.1 Data Splitting and Cross-Validation
</h3>
Stratified K-Fold Cross-Validation
Train-Test Split
<h3>
3.2 Model Selection and Training
</h3>
Boosting Models
Bayesian Models
K-Nearest Neighbors (KNN)
Logistic Regression
Fitting and Training Models
<h3>
3.3 Hyperparameter Tuning
</h3>
Pruning Decision Trees
Visualizing Pruning
Model Tuning (GridSearchCV)
<h3>
3.4 Model Evaluation
</h3>
Plotting Evaluation Scores
<h3>
3.5 Ensemble Models
</h3>
Stacking Classifier
Voting Classifier
  <h2>
Section 4: Pipeline Creation
</h2><h3>
4.1 Preprocessing Pipeline
</h3>
Preprocessor Functions (preprocessor.py)
Training Preprocessor Pipeline (train_preprocessor.pkl)

  <h3>4.2 Testing Pipeline
</h3>
Testing Preprocessor Functions (test_preprocessor.py)
Testing Preprocessor Pipeline (test_preprocessor.pkl)

  <h3>4.3 Model Pipeline
</h3>
Building and Saving Model Pipeline (model.pkl)
Applying Pipelines for Fitting, Training, and Evaluation

