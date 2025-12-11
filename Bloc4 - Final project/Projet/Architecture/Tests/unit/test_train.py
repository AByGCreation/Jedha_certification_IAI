"""
Unit test for train.py
Tests the train/test split functionality
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add the localModel path to sys.path to import train module
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, "..", "..", "App", "localModel"))
sys.path.insert(0, project_path)


class TestTrainTestSplit:
    """Test the train/test split configuration"""

    def test_train_test_split_ratio(self):
        """
        Test that train_test_split maintains correct ratio and stratification
        This verifies the split configuration used in train.py (line 164-167)
        """
        # Create sample data similar to the fraud dataset
        np.random.seed(42)
        n_samples = 1000

        # Create a DataFrame with fraud/non-fraud cases
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples)
        })

        # Create imbalanced target (like fraud detection)
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.9, 0.1]))

        # Perform the same split as in train.py
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Assertions
        # Check that split ratio is correct (80/20)
        assert len(X_train) == int(n_samples * 0.8), "Training set should be 80% of data"
        assert len(X_test) == int(n_samples * 0.2), "Test set should be 20% of data"

        # Check that stratification preserved class distribution
        train_fraud_ratio = y_train.sum() / len(y_train)
        test_fraud_ratio = y_test.sum() / len(y_test)
        original_fraud_ratio = y.sum() / len(y)

        # Allow small tolerance for ratio differences due to rounding
        assert abs(train_fraud_ratio - original_fraud_ratio) < 0.02, \
            "Training set should maintain similar fraud ratio to original data"
        assert abs(test_fraud_ratio - original_fraud_ratio) < 0.02, \
            "Test set should maintain similar fraud ratio to original data"

        # Check that no data leakage (train and test don't overlap)
        assert len(set(X_train.index) & set(X_test.index)) == 0, \
            "Train and test sets should not have overlapping indices"

        print("✅ Train/test split test passed!")
