# AML Detection with Advanced Feature Selection and Stacking Ensemble
## 1. Project Overview
This project aims to build a high-performance AML detection model by integrating an advanced feature selection process with many ML models (especially the boosting models) and a self-defined multi-model stacking ensemble. The core idea is to first reduce feature redundancy and enhance interpretability using the advanced feature selection technique which contains the idea of hierarchy clustering, VIF, and a score function combines the feature's variance and its absolute correlation with the target to select the cluster's representative, and then leverage the complementary strengths of diverse base models through a robust stacking strategy. By combining Out-Of-Fold probability outputs from the base learners with the original features, our custom stacking ensemble—designed in a modular and extensible class-based framework superior predictive accuracy and improved generalization performance.

## 2. Project File Structure
<pre>
├── README.md            
├── featureSelector.py    
├── stackingEnsemble.py  
└── Modeling.ipynb       
</pre>
* **README.md**: Provides an overall introduction, explanation of the modules, usage instructions, and potential extensions.
* **featureSelector**.py: Contains the FeatureSelector class that reduces feature redundancy using hierarchical clustering, VIF checking, and a representative feature selection strategy.
* **stackingEnsemble.py**: Contains the StackingEnsemble class, which implements the stacking ensemble pipeline. This includes generating OOF predictions from base models, combining these with original features for meta-model training, optimizing the decision threshold, and performing detailed model evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix etc.).
* **Modeling.ipynb**: A detailed Jupyter Notebook that demonstrates how to load and preprocess data, apply feature selection, build the ML models and the stacking ensemble, and evaluate the results. Additionally, the detailed features description is also included.

## 3. Data description
The dataset used in this project comprise detailed blockchain transaction records extracted from public ledgers on the TRON network. The data includes both illicit wallet addresses—identified and labeled through law enforcement databases and reported fraudulent cases—and a set of regular wallet addresses, which were randomly sampled from the network to serve as a control group. In total, the dataset contains around **15,262** wallet addresses and a comprehensive set of **82** features. In total, these features are categorized into four groups: Wallet Profile Features (e.g., lifetime, total transactions, balances), Transaction Amount Features (e.g., average, max, min amounts, extreme transactions), Transaction Temporal Features (e.g., time gaps, skewness, kurtosis), and TRON-Specific Features (e.g., TRX and USDT-related metrics). This comprehensive dataset forms the basis for our AML detection model.

## 4. Class Description
As the highlight of this project, the ideas of selecting features and stacking ensemble are transformed to FeatureSelector Class and StackingEnsemble Class respectively.

### 4.1 FeatureSelector Class
The **FeatureSelector Class** automates the feature selection process to mitigate high dimensionality and feature redundancy. It follows these steps:
* **Compute Distance Matrix**:
Calculates the absolute correlation between features and converts it into a distance matrix using the formula:

   $\text{distance} = 1 - |\text{correlation}|$ .
* **Hierarchical Clustering**:
Uses complete linkage to group features with high correlations (above a specified threshold) into clusters.
* **VIF Checking**:
Within each cluster, it recursively removes features with high multicollinearity by computing the Variance Inflation Factor (VIF).
* **Representative Feature Selection**:
In each cluster, a scoring function is applied that combines the feature’s variance and its absolute correlation with the target. The feature with the highest score is selected as the cluster’s representative with the formula:

    $\arg \max _{X_i \in C}\left(\alpha Var\left(X_i\right)+(1-\alpha)\left|\rho\left(X_i, y\right)\right|\right)$.
* Final Output:
Returns a list of selected, representative features that are more independent and informative.

### 4.2 StackingEnsemble Class
The **StackingEnsemble Class** implements a multi-model stacking framework with several key components:

* **Base Model OOF Predictions**:
Each base model is trained using KFold cross-validation to generate Out-Of-Fold (OOF) probability predictions. This ensures that each sample’s predicted probability is obtained from a model that did not see that sample during training, reducing overfitting.
* **Enhanced Meta-Features**:
The OOF probabilities are concatenated with the original features to create an enhanced feature matrix for the meta-model.
* **Meta-Model Training**:
The meta-model is trained on the enhanced meta-features. The class supports automatic threshold optimization via nested cross-validation and can be extended to multi-layer stacking if desired.
* **Threshold Optimization**:
The optimize_threshold method performs nested cross-validation on the meta-training data to automatically select the decision threshold that maximizes a chosen metric (e.g., weighted F1-score).
* **Evaluation**:
The evaluate method computes key metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC), prints a detailed classification report, and plots the confusion matrix.


