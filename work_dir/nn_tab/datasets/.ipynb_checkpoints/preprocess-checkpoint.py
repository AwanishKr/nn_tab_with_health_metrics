import pandas as pd
import numpy as np 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn
import pickle
import os


class RobustScaleSmoothClipTransform(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, X, y=None, feature_columns=None):
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            if feature_columns is None:
                feature_columns = X.columns.tolist()
            
            self.feature_columns_ = feature_columns
            X = X[feature_columns].values.astype(np.float32)
        else:
            self.feature_columns_ = None
        
        self._median = np.median(X, axis=-2)
        self.quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        
        self.max = np.max(X, axis=-2)
        self.min = np.min(X, axis=-2)
        idxs = self.quant_diff == 0.0
        
        # on indexes where the quantile difference is zero, do min-max scaling instead
        self.quant_diff[idxs] = 0.5 * (self.max[idxs] - self.min[idxs])
        factors = 1.0 / (self.quant_diff + 1e-30)
        
        # if feature is constant on the training data, set factor to zero so that it is also constant at prediction time
        factors[self.quant_diff == 0.0] = 0.0
        self._factors = factors
        
        # Create parameters directory if it doesn't exist
        params_dir = os.path.join(os.getcwd(), "parameters")
        os.makedirs(params_dir, exist_ok=True)
        
        # Save transformer parameters to current directory
        params_file = os.path.join(params_dir, "transformer_params.pkl")
        with open(params_file, 'wb') as f:
            pickle.dump({"median": self._median, "factors": self._factors}, f)
        return self

    def transform(self, X, y=None, feature_columns=None):
        if isinstance(X, pd.DataFrame):
            # Use feature columns from fit if not specified
            if feature_columns is None:
                feature_columns = self.feature_columns_
            
            # Create a copy to avoid modifying original
            X_transformed = X.copy()
            
            # Extract and transform only feature columns
            X_features = X[feature_columns].values.astype(np.float32)
            X_features_transformed = self._factors[None, :] * (X_features - self._median[None, :])
            X_features_transformed = X_features_transformed / np.sqrt(1 + (X_features_transformed / 3) ** 2)
            
            # Replace feature columns with transformed values
            X_transformed[feature_columns] = X_features_transformed
            return X_transformed
        else:
            X = self._factors[None, :] * (X - self._median[None, :])
            return X / np.sqrt(1 + (X / 3) ** 2)


def clean_date(x):
    # If it's a bytes object, decode it.
    if isinstance(x, bytes):
        return x.decode('utf-8')
    
    # If it's a string that starts with "b'" and ends with "'", remove those parts.
    elif isinstance(x, str) and x.startswith("b'") and x.endswith("'"):
        return x[2:-1]
    return x       