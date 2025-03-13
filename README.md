# Dicoding-Building-Machine-Learning-Projects
This repository contains Machine Learning projects from the Building Machine Learning Projects class in Dicoding.

## 1. Machine Learning Workflow
A Machine Learning Workflow is the steps taken to build a good and optimal ML model. Here are the general steps:

1. **Problem Understanding**: Determine the goals and problems to be solved.
2. **Data Collection**: collect data from various sources.
3. **Data Preprocessing**: Clean and transform the data so that it is ready to use.
4. **Feature Engineering**: Create new features or select relevant features.
5. **Model Selection**: Choose an appropriate algorithm.
6. **Training Model**: Train the model using the training data.
7. **Evaluating Model**: Use evaluation metrics to measure model performance.
8. **Optimizing Model**: Perform hyperparameter tuning to improve performance.
9. **Deploy and Monitor**: Implement the model into the system and monitor its performance.

---
## 2. Supervised Learning - Classification
Classification is a type of supervised learning that aims to predict the category or class of data.

### Commonly Used Classification Algorithms:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbor (KNN)**
- **Neural Network**

### Assessment Metrics:
- **Accuracy**
- **Precision, Recall, and F1-Score**
- **Confusion Matrix**
- **ROC-AUC Score**

---
## 3. Supervised Learning - Regression
Regression is used to predict numeric values ​​based on input variables.

### Commonly Used Regression Algorithms:
- **Linear Regression**
- **Polynomial Regression**
- **Ridge Regression & Lasso Regression**
- **Decision Tree Regression**
- **Random Forest Regression**
- **Support Vector Regression (SVR)**

### Assessment Metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared (R²)**

---
## 4. Unsupervised Learning - Clustering
Clustering is a technique in unsupervised learning that groups data based on similar patterns.

### Commonly Used Clustering Algorithms:
- **K-Means Clustering**
- **Hierarchical Clustering**
- **DBSCAN (Density-Based Clustering of Spatial Applications with Noise)**
- **Gaussian Mixture Model (GMM)**

### Assessment Metrics:
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Elbow Method (for K-Means)**

---
## 5. Feature Engineering Techniques
Feature Engineering is the process of creating new features or modifying existing features to improve model performance.

### Some Feature Engineering Techniques:
- **Feature Scaling (MinMaxScaler, StandardScaler)**
- **One-Hot Encoding & Label Encoding**
- **Feature Selection (PCA, Mutual Information, Recursive Feature Elimination)**
- **Feature Extraction (Word Embeddings, TF-IDF for text)**

---
## 6. Overfitting and Underfitting
### Overfitting:
- Model is too complex so it learns from noisy data.
- Characteristics: High performance on training set, but poor on test set.
- Ways to overcome:
- Regularization (L1/L2)
- Pruning on Decision Tree
- Reducing the number of features
- Increasing data with Augmentation

### Underfitting:
- Model is too simple so it doesn't capture enough patterns from the data.
- Characteristics: Poor performance on both training and test sets.
- How to overcome:
- Add more relevant features
- Increase model complexity
- Use more flexible algorithms

---
## 7. Optimization Model with Hyperparameter Tuning
Hyperparameter tuning is the process of finding the best combination of parameters to improve model performance.

### Hyperparameter Tuning Methods:
- **Grid Search**: Trying all combinations of specified parameters.
- **Random Search**: Trying random combinations of parameters.
- **Bayesian Optimization**: Using a probabilistic model to find optimal parameters.
- **Hyperband & Optuna**: More efficient methods with iteration-based evaluation.

---
