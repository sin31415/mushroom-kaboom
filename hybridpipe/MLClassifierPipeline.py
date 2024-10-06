# MLClassifier.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
import matplotlib.pyplot as plt
import seaborn as sns

class MLClassifierPipeline:
    def __init__(
        self,
        df,
        target_column,
        test_size=0.2,
        random_state=42,
        display_analytics=True,
    ):
        """
        Initializes the ML pipeline.

        Parameters:
        - df: pandas DataFrame containing the dataset.
        - target_column: The name of the target column.
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Controls the shuffling applied to the data before applying the split.
        - display_analytics: Whether to display analytics for each model.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.display_analytics = display_analytics
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Extra Trees': ExtraTreesClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier(),
        }
        self.model_results = {}
        self.best_model = None
        self.preprocessor = None
        self.trained_models = {}
        self.user_model = None

    def preprocess_data(self, drop_max_na_col=False, drop_threshold=0.25):
        """Cleans and preprocesses the data."""

        #Drops column with maximum nulls
        if drop_max_na_col==True:
            #Check for features with 25% or more NaNs and drop them
            missing_percentage = self.df.isnull().mean()
            features_to_drop = missing_percentage[missing_percentage >= drop_threshold].index
            self.df.drop(columns=features_to_drop, inplace=True)
            print(f"Dropped features with >=25% missing values: {features_to_drop.tolist()}")
            if (len(features_to_drop.tolist())/len(df.columns)) > 0.5:
                print(f"WARNING: More than 50% of the columns have been dropped per your threshold ({drop_threshold}).")
        
        # Drop rows with null values
        self.df.dropna(inplace=True)
        print(f"{len(df)} number of instances remaining.")

        # Separate features and target
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Define preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ]
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Fit and transform the training data, transform the test data
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_models(self):
        """Trains all models and evaluates them."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            clf = Pipeline(steps=[('model', model)])
            clf.fit(self.X_train, self.y_train)
            y_pred_train = clf.predict(self.X_train)
            y_pred_test = clf.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)

            self.model_results[name] = {
                'model': clf,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'y_pred_test': y_pred_test,
            }

            if self.display_analytics:
                print(f"\n{name} Results:")
                print(f"Training Accuracy: {train_accuracy:.4f}")
                print(f"Test Accuracy: {test_accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(self.y_test, y_pred_test))
                print("Confusion Matrix:")
                cm = confusion_matrix(self.y_test, y_pred_test)
                print(cm)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, cbar_kws={'label': 'Scale'}, vmin=0, vmax=705)
                plt.title(f'{name} Confusion Matrix')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                plt.tight_layout()
                plt.show()

            # Store trained model
            self.trained_models[name] = clf

    def select_best_model(self):
        """Selects the best model based on test accuracy."""
        best_accuracy = 0
        best_model_name = None

        for name, results in self.model_results.items():
            if results['test_accuracy'] > best_accuracy:
                best_accuracy = results['test_accuracy']
                best_model_name = name

        self.best_model = self.model_results[best_model_name]['model']
        print(f"The best model is {best_model_name} with a test accuracy of {best_accuracy:.4f}.")

    def visualize_decision_tree(self):
        """Visualizes the trained Decision Tree model."""
        # Check if the Decision Tree has been trained
        if 'Decision Tree' in self.trained_models:
            model = self.trained_models['Decision Tree'].named_steps['model']

            # Get the original class names from the target column (e.g., 'e' and 'p')
            class_names = self.y_train.unique()

            # Get the feature names from the preprocessor
            feature_names = self.preprocessor.get_feature_names_out()

            plt.figure(figsize=(40, 20))
            plot_tree(model, feature_names=feature_names, filled=True, rounded=True, class_names=class_names)
            plt.title('Decision Tree Visualization')
            plt.show()
        else:
            print("Decision Tree model is not trained yet.")

    def get_feature_importance(self):
        """Displays feature importance for models that support it."""

        if self.best_model is None and self.user_model is None:
            print("Please run select_best_model() or user_model_return() first.")
            return
    
        if self.user_model is not None:
            model = self.user_model.named_steps['model']
        elif self.best_model is not None:
            model = self.best_model.named_steps['model']
            
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.preprocessor.get_feature_names_out()
            feature_importances = pd.Series(importances, index=feature_names)
            feature_importances.sort_values(ascending=False, inplace=True)
            plt.figure(figsize=(10, 6))
            feature_importances.head(20).plot(kind='bar')
            plt.title('Feature Importances')
            plt.show()
        else:
            print(f"The model {model} does not support feature importance.")
            

    def user_model_return(self):
        if self.trained_models:
            keys_list = list(self.trained_models.keys())
            print('The models trained are: \n')
            for i, name in enumerate(keys_list):
                print(f"{i}. {name}")
            choice = int(input('Which model would you like?'))

            # Access the value by index
            self.user_model = self.trained_models[keys_list[choice]]
            return self.user_model.named_steps['model']
        else:
            print('No models are trained yet. Please train the models to choose.')
    
    def run_pipeline(self, drop_max_na_col_in=False, drop_threshold_in=True):
        """Runs the full pipeline."""
        
        self.preprocess_data(drop_max_na_col=drop_max_na_col_in, drop_threshold=drop_threshold_in)
        self.train_models()
        self.select_best_model()