# -*- coding: utf-8 -*-
class StackingEnsemble:
    """
    Attributes:
        base_models: the list of (name, model_instance) for base models
        meta_model: the meta-model used to combine base model predictions
        n_splits: the num of folds for KFold
        threshold: the decision threshold for final predictions (default 0.5, can be optimized)
        random_state: random seed for reproducibility
    """
    def __init__(self, base_models, meta_model, n_splits=5, threshold=0.5, random_state=42):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_splits = n_splits
        self.threshold = threshold
        self.random_state = random_state
        self.trained_base_models = []
        self.oof_predictions = None
        self.y_train = None

    #  generate OOF predictions using KFold cross-validation for a given base model.
    def get_oof_predictions(self, model, X, y):
        """
        Parameters:
            model: a base model instance.
            X: the selected training features.
            y: the training labels.

        Returns:
            oof_preds: the array of OOF positive class probabilities with shape (n_samples,).
            fitted_model: the model re-fitted on the entire training set.
        """

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y

        n_samples = X_array.shape[0]
        oof_preds = np.zeros(n_samples, dtype=float)
        kf = KFold(n_splits = self.n_splits, shuffle = True, random_state = self.random_state)

        for train_idx, val_idx in kf.split(X_array, y_array):
            X_train_fold, X_val_fold = X_array[train_idx], X_array[val_idx]
            y_train_fold, y_val_fold = y_array[train_idx], y_array[val_idx]
            model.fit(X_train_fold, y_train_fold)
            oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]

        # refit the model on the entire training set for later test prediction
        model.fit(X_array, y_array)
        return oof_preds, model

    def fit(self, X_train, y_train):
        """
        Train the stacking ensemble:
            1. For each base model, generate the OOF predictions via KFold CV
            2. Stack these OOF probabilities with original features to create the enhanced meta features
            3. Fit the meta-model on these enhanced features

        Parameters:
            X_train: the selected training features
            y_train: the training labels
        """
        self.y_train = np.array(y_train)
        oof_preds_list = []
        self.trained_base_models = []

        print("Generating OOF predictions for base models...")
        for name, mdl in self.base_models:
            print(f"Processing base model: {name}")
            oof_probs, fitted_mdl = self.get_oof_predictions(mdl, X_train, y_train)
            oof_preds_list.append(oof_probs.reshape(-1, 1))
            self.trained_base_models.append((name, fitted_mdl))

        # stack OOF predictions with shape = (n_samples, n_base_models)
        stack_X_proba = np.hstack(oof_preds_list)
        # print("Stacked OOF probabilities shape:", stack_X_proba.shape)

        # obtain original features as a numpy array
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train

        # concatenate OOF probabilities with the original selected features
        self.oof_predictions = np.hstack([stack_X_proba, X_train_array])
        # print("Enhanced stacking training feature shape:", self.oof_predictions.shape)

        # train the meta-model on the enhanced features
        self.meta_model.fit(self.oof_predictions, y_train)
        # print("Meta-model training complete.")

    # generate the final predictions on the test set
    def predict(self, X_test):
        """
        Parameters:
            X_test : the test features

        Returns:
            final_probs : the predicted positive class probabilities
            final_preds : the binary predictions based on the decision threshold
        """
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test

        test_preds_list = []
        for name, mdl in self.trained_base_models:
            proba_test = mdl.predict_proba(X_test_array)[:, 1]
            test_preds_list.append(proba_test.reshape(-1, 1))

        # stack the base model test probabilities
        stack_test_proba = np.hstack(test_preds_list)
        print("Stacked test probabilities shape:", stack_test_proba.shape)

        # concatenate with the original selected test features
        stack_test_enhanced = np.hstack([stack_test_proba, X_test_array])
        print("Enhanced stacking test feature shape:", stack_test_enhanced.shape)

        # get the meta-model probabilities
        final_probs = self.meta_model.predict_proba(stack_test_enhanced)[:, 1]
        final_preds = (final_probs > self.threshold).astype(int)
        return final_probs, final_preds

    # optimize the decision threshold for converting meta-model probabilities to binary predictions
    # use KFold cross-validation on the meta training features and training labels.
    def optimize_threshold(self, thresholds = np.linspace(0.3, 0.7, 100), cv = 5):
        """
        Parameters:
            thresholds: candidate threshold values (default: 0.3 to 0.7)
            cv: number of folds for threshold optimization (default: 5)

        Returns:
            best_threshold: the threshold value that maximizes the weighted F1-score
        """
        if self.oof_predictions is None or self.y_train is None:
            raise ValueError("Please call the fit() method first to generate meta training features and store y_train.")

        best_threshold = 0.5
        best_score = -1
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        # use the meta-model's predictions on the meta training data for optimization
        meta_preds = self.meta_model.predict_proba(self.oof_predictions)[:, 1]

        for thresh in thresholds:
            scores = []
            for train_idx, val_idx in kf.split(self.oof_predictions):
                y_val = self.y_train[val_idx]
                preds_val = (meta_preds[val_idx] > thresh).astype(int)
                scores.append(f1_score(y_val, preds_val, average='weighted'))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = thresh

        self.threshold = best_threshold
        print(f"Optimized threshold: {best_threshold:.4f}, with CV weighted F1 score: {best_score:.4f}")
        return best_threshold

    # evaluate the stacking ensemble on the test set, outputting Accuracy, Precision, Recall, F1-score,
    # and plotting the confusion matrix
    def evaluate(self, X_test, y_test, num_classes=2):
        """
        Parameters:
            X_test: the selected test features
            y_test: the true test labels
            num_classes: Number of classes (default=2)

        Returns:
            metrics: the dictionary containing accuracy, precision, recall, f1_score, and ROC-AUC
            y_pred: the final predicted binary labels
        """

        final_probs, final_preds = self.predict(X_test)

        accuracy = accuracy_score(y_test, final_preds)
        precision = precision_score(y_test, final_preds, average='weighted')
        recall = recall_score(y_test, final_preds, average='weighted')
        f1 = f1_score(y_test, final_preds, average='weighted')
        roc_auc_val = roc_auc_score(y_test, final_probs)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, final_preds))

        # Plot confusion matrix
        cm = confusion_matrix(y_test, final_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Prediction")
        plt.ylabel("True Value")
        plt.show()

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc_val
        }
        return metrics, final_preds
