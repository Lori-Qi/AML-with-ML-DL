# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FeatureSelector:
    """
    parameters:
    df: pd.DataFrame
        Dataframe containing the selected features
    y: array-like or pd.Series
        The traget variable which used to compute the correlation between the features and the target
    """

    def __init__(self, df, y):
        self.df = df.copy()
        self.y = np.array(y)

    # correlation distance matrix between the features, d = 1 - abs(corr)
    def compute_distance_matrix(self):
        corr = self.df.corr().abs()
        dist = 1 - corr
        return dist

    # implementing hierarchical clustering using complete linkage to classify the features into many clusters
    def hierarchical_clustering(self, dist_matrix, threshold=0.8):
        """
        Parameters:
        dist_matrix : pd.DataFrame
            The distance matrix between features
        threshold : float, optional
            The correlation threshold (between 0 and 1). Features with correlation higher than this threshold
            are considered highly correlated. In clustering, the cut-off distance is (1 - threshold)

        Returns:
        cluster_dict : dict
            A dictionary of clusters in the format {cluster_id: [feature1, feature2, ...]}
        """
        condensed_dist = sch.distance.squareform(dist_matrix.values)
        linkage_matrix = sch.linkage(condensed_dist, method='complete')
        clusters = sch.fcluster(linkage_matrix, t=(1 - threshold), criterion='distance')
        cluster_dict = {}
        features = dist_matrix.columns
        for feature, c_id in zip(features, clusters):
            cluster_dict.setdefault(c_id, []).append(feature)
        return cluster_dict

    #  calculate the VIF for each feature in the given DataFrame
    def claculate_vif(self, df):
        """
        Parameters:
        df : pd.DataFrame
            The subset of features for which to calculate VIF

        Returns:
        vif_df : pd.DataFrame
            A DataFrame containing each feature and its corresponding VIF value
        """
        X = df.values
        vif_data = []
        for i in range(X.shape[1]):
            vif = variance_inflation_factor(X, i)
            vif_data.append(vif)
        vif_df = pd.DataFrame({'feature': df.columns, 'vif': vif_data})
        return vif_df

        # recursively remove features with VIF higher than the specified threshold until all features have VIF
        # below the threshold or only one feature remains.
        def_remove_high_vif_features(self, df, vif_threshold = 10.0):
            """
            Parameters:
            df : pd.DataFrame
                The subset of features to be checked for VIF
            vif_threshold : float, optional
                The VIF threshold (default value is 10.0)

            Returns:
            df_reduced : pd.DataFrame
                The DataFrame after removing high-VIF features
            """
            while True:
                if df.empty or df.shape[1] <= 1:
                    break  # exit if DataFrame is empty or only one column remains.
                vif_df = self.calculate_vif(df)
                max_vif = vif_df['vif'].max()
                if max_vif >= vif_threshold and df.shape[1] > 1:
                    drop_feat = vif_df.loc[vif_df['vif'].idxmax(), 'feature']
                    df = df.drop(columns=[drop_feat])
                else:
                    break
            return df

        # select the most representative feature from a cluster of features
        # the score is computed as: score = alpha * Var(feature) + (1 - alpha) * |corr(feature, y)|
        def selecte_representative_feature(self, df_cluster, alpha = 0.3):
            """
            Parameters:
            df_cluster : pd.DataFrame
                The subset of features in a cluster
            alpha : float, optional
                The weight for the variance term. Default is 0.3

            Returns:
            best_feat : str
                The name of the selected representative feature
            """
            if df_cluster.shape[1] == 1:
                return df_cluster.columns[0]

            var_series = df_cluster.var()
            corr_with_y = {}
            for col in df_cluster.columns:
                corr_with_y[col] = abs(np.corrcoef(df_cluster[col].values, self.y)[0, 1])

            best_feat = None
            best_score = -1
            for col in df_cluster.columns:
                score = alpha * var_series[col] + (1 - alpha) * corr_with_y[col]
                if score > best_score:
                    best_score = score
                    best_feat = col
            return best_feat

        # combine hierarchical clustering, VIF checking, and representative feature selection
        # to obtain a set of representative features.

        def improved_hier_cluster_feature_selection(self, corr_threshold=0.9, vif_threshold=10.0, alpha=0.5):
            """
            Parameters:
            corr_threshold : float, optional
            The correlation threshold used in hierarchical clustering (default 0.9)
            vif_threshold : float, optional
                The VIF threshold for feature removal (default 10.0)
            alpha : float, optional
                The weight for variance in the scoring function (default 0.5)

            Returns:
            representative_features : list
                A list of the names of the selected representative features
            """

            dist_matrix = self.compute_distance_matrix()
            cluster_dict = self.hierarchical_clustering(dist_matrix, threshold=corr_threshold)
            representative_features = []
            for c_id, feature_list in cluster_dict.items():
                df_subset = self.df[feature_list].copy()
                df_subset_vif_checked = self.remove_high_vif_features(df_subset, vif_threshold=vif_threshold)
                best_feat = self.select_representative_feature(df_subset_vif_checked, alpha=alpha)
                representative_features.append(best_feat)
            return representative_features
