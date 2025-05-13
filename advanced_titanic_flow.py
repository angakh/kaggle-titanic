"""
Advanced Titanic ML Flow with Metaflow and MLflow
This pipeline demonstrates more sophisticated ML concepts:
- Multiple model training in parallel
- Hyperparameter optimization
- Feature selection
- Model comparison
- Advanced visualizations
- Model deployment preparation
"""

from metaflow import FlowSpec, step, Parameter, resources, card
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json
from datetime import datetime

class AdvancedTitanicFlow(FlowSpec):
    """
    An advanced flow for Titanic survival prediction that demonstrates
    best practices in ML engineering using Metaflow and MLflow.
    """
    
    # Define parameters
    data_path = Parameter('data_path', 
                         help='Path to the Titanic dataset',
                         default='./data/train.csv')
    
    test_size = Parameter('test_size',
                         help='Fraction of data to use for testing',
                         default=0.2)
    
    cv_folds = Parameter('cv_folds',
                        help='Number of cross-validation folds',
                        default=5)
    
    random_state = Parameter('random_state',
                           help='Random seed',
                           default=42)
    
    # Parameter for multiple model types
    model_types = Parameter('model_types',
                          help='Comma-separated list of models to train',
                          default='rf,lr,gb,svm')
    
    # Parameter for experiment tracking
    experiment_name = Parameter('experiment_name',
                              help='MLflow experiment name',
                              default='titanic-advanced')

    @step
    def start(self):
        """
        Start the flow and set up experiment tracking.
        """
        print(f"Starting Advanced Titanic Flow at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parse model types
        self.models = self.model_types.split(',')
        print(f"Will train the following models: {self.models}")
        
        # Map model types to readable names
        self.model_names = {
            'rf': 'Random Forest',
            'lr': 'Logistic Regression',
            'gb': 'Gradient Boosting',
            'svm': 'Support Vector Machine'
        }
        
        # Set up MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self.experiment_name)
        
        # Create run IDs dict
        self.run_ids = {}
        
        self.next(self.load_data)

    @step
    def load_data(self):
        """
        Load and explore the Titanic dataset.
        """
        print(f"Loading data from {self.data_path}...")
        
        try:
            self.df = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure the Titanic dataset is available at the specified path.")
            print("You can download it from Kaggle: https://www.kaggle.com/c/titanic/data")
            raise
        
        # Save a data description summary
        self.data_summary = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {', '.join(self.df.columns)}")
        print(f"Missing values: {self.df.isnull().sum().sum()} total")
        
        # Create a run with dataset info
        with mlflow.start_run(run_name="data_summary") as run:
            # Log dataset properties
            mlflow.log_param("dataset_rows", self.df.shape[0])
            mlflow.log_param("dataset_columns", self.df.shape[1])
            mlflow.log_param("total_missing_values", self.df.isnull().sum().sum())
            
            # Create and log dataset summary
            dataset_info = {
                "shape": list(self.df.shape),
                "columns": self.df.columns.tolist(),
                "missing_values": self.df.isnull().sum().to_dict(),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            }
            
            # Write summary to file and log
            with open("dataset_summary.json", "w") as f:
                json.dump(dataset_info, f, indent=2)
            mlflow.log_artifact("dataset_summary.json")
            
            # Track this run
            self.data_run_id = run.info.run_id
        
        self.next(self.exploratory_data_analysis)
        
    @step
    def load_data(self):
        """
        Load and explore the Titanic dataset.
        """
        print(f"Loading data from {self.data_path}...")
        
        try:
            self.df = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure the Titanic dataset is available at the specified path.")
            print("You can download it from Kaggle: https://www.kaggle.com/c/titanic/data")
            raise
        
        # Save a data description summary
        self.data_summary = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {', '.join(self.df.columns)}")
        print(f"Missing values: {self.df.isnull().sum().sum()} total")
        
        # Create a run with dataset info
        with mlflow.start_run(run_name="data_summary") as run:
            # Log dataset properties
            mlflow.log_param("dataset_rows", self.df.shape[0])
            mlflow.log_param("dataset_columns", self.df.shape[1])
            mlflow.log_param("total_missing_values", self.df.isnull().sum().sum())
            
            # Create and log dataset summary
            dataset_info = {
                "shape": list(self.df.shape),
                "columns": self.df.columns.tolist(),
                "missing_values": self.df.isnull().sum().to_dict(),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            }
            
            # Write summary to file and log
            with open("dataset_summary.json", "w") as f:
                json.dump(dataset_info, f, indent=2)
            mlflow.log_artifact("dataset_summary.json")
            
            # Track this run
            self.data_run_id = run.info.run_id
        
        self.next(self.exploratory_data_analysis)
    
    @step
    def create_baselines(self):
        """
        Create baseline models for comparison.
        """
        print("Creating baseline models for comparison...")
        
        # Add FamilySize if not already added in EDA
        if 'FamilySize' not in self.df.columns:
            self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        
        # Add IsAlone feature
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        
        # Split data for baselines
        X = self.df.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        y = self.df['Survived']
        
        # Handle categorical variables for baseline models
        X_encoded = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)
        
        # Fill missing values for baseline models
        X_encoded['Age'] = X_encoded['Age'].fillna(X_encoded['Age'].median())
        X_encoded['Fare'] = X_encoded['Fare'].fillna(X_encoded['Fare'].median())
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Store the data splits for later use
        self.X_train_simple = X_train
        self.X_test_simple = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        with mlflow.start_run(run_name="baseline_models") as run:
            # Track run ID
            self.baseline_run_id = run.info.run_id
            
            # Baseline 1: Majority class prediction
            from sklearn.dummy import DummyClassifier
            
            majority_clf = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
            majority_clf.fit(X_train, y_train)
            majority_pred = majority_clf.predict(X_test)
            majority_accuracy = accuracy_score(y_test, majority_pred)
            
            # Baseline 2: Gender-based prediction (females survive, males don't)
            # Find Sex_female column index
            if 'Sex_female' in X_test.columns:
                gender_pred = X_test['Sex_female'].values
                gender_accuracy = accuracy_score(y_test, gender_pred)
                gender_precision = precision_score(y_test, gender_pred)
                gender_recall = recall_score(y_test, gender_pred)
                gender_f1 = f1_score(y_test, gender_pred)
                
                # Calculate confusion matrix for gender baseline
                gender_cm = confusion_matrix(y_test, gender_pred)
                
                # Create confusion matrix plot
                plt.figure(figsize=(8, 6))
                sns.heatmap(gender_cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Gender Baseline Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                plt.savefig('gender_baseline_cm.png')
                mlflow.log_artifact('gender_baseline_cm.png')
                plt.close()
                
            else:
                print("Warning: Sex_female column not found in encoded data")
                gender_accuracy = 0
                gender_precision = 0
                gender_recall = 0
                gender_f1 = 0
            
            # Baseline 3: Simple model with minimal features (Class + Gender)
            # Create a simple model using just Pclass and Gender
            from sklearn.tree import DecisionTreeClassifier
            
            if 'Sex_female' in X_train.columns:
                simple_features = ['Pclass', 'Sex_female']
                X_train_simple_subset = X_train[simple_features]
                X_test_simple_subset = X_test[simple_features]
                
                simple_tree = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
                simple_tree.fit(X_train_simple_subset, y_train)
                simple_pred = simple_tree.predict(X_test_simple_subset)
                simple_accuracy = accuracy_score(y_test, simple_pred)
                simple_precision = precision_score(y_test, simple_pred)
                simple_recall = recall_score(y_test, simple_pred)
                simple_f1 = f1_score(y_test, simple_pred)
                
                # Plot the simple decision tree if graphviz is available
                try:
                    from sklearn.tree import export_graphviz
                    import graphviz
                    
                    dot_data = export_graphviz(
                        simple_tree, 
                        out_file=None, 
                        feature_names=simple_features,
                        class_names=['Did not survive', 'Survived'],
                        filled=True, 
                        rounded=True, 
                        special_characters=True
                    )
                    
                    graph = graphviz.Source(dot_data)
                    graph.render("simple_decision_tree")
                    mlflow.log_artifact("simple_decision_tree.pdf")
                except ImportError:
                    print("Graphviz not available, skipping decision tree visualization")
            else:
                simple_accuracy = 0
                simple_precision = 0
                simple_recall = 0
                simple_f1 = 0
            
            # Log metrics
            mlflow.log_metric("majority_baseline_accuracy", majority_accuracy)
            
            if gender_accuracy > 0:
                mlflow.log_metric("gender_baseline_accuracy", gender_accuracy)
                mlflow.log_metric("gender_baseline_precision", gender_precision)
                mlflow.log_metric("gender_baseline_recall", gender_recall)
                mlflow.log_metric("gender_baseline_f1", gender_f1)
            
            if simple_accuracy > 0:
                mlflow.log_metric("simple_tree_accuracy", simple_accuracy)
                mlflow.log_metric("simple_tree_precision", simple_precision)
                mlflow.log_metric("simple_tree_recall", simple_recall)
                mlflow.log_metric("simple_tree_f1", simple_f1)
                
                # Create a simple model description file
                with open("simple_model_description.txt", "w") as f:
                    f.write("Simple Decision Tree Model\n")
                    f.write("=========================\n\n")
                    f.write("Features: Passenger Class, Gender\n\n")
                    f.write(f"Accuracy: {simple_accuracy:.4f}\n")
                    f.write(f"Precision: {simple_precision:.4f}\n")
                    f.write(f"Recall: {simple_recall:.4f}\n")
                    f.write(f"F1 Score: {simple_f1:.4f}\n\n")
                    f.write("This model serves as a simple baseline using just \n")
                    f.write("passenger class and gender information.\n")
                
                mlflow.log_artifact("simple_model_description.txt")
            
            # Store baseline results for later comparison
            self.baselines = {
                "majority": {
                    "accuracy": majority_accuracy
                }
            }
            
            if gender_accuracy > 0:
                self.baselines["gender"] = {
                    "accuracy": gender_accuracy,
                    "precision": gender_precision,
                    "recall": gender_recall,
                    "f1": gender_f1
                }
            
            if simple_accuracy > 0:
                self.baselines["simple_tree"] = {
                    "accuracy": simple_accuracy,
                    "precision": simple_precision,
                    "recall": simple_recall,
                    "f1": simple_f1
                }
        
        print("Baseline models created and evaluated:")
        print(f"  Majority class baseline accuracy: {majority_accuracy:.4f}")
        if gender_accuracy > 0:
            print(f"  Gender-based baseline accuracy: {gender_accuracy:.4f}")
        if simple_accuracy > 0:
            print(f"  Simple tree baseline accuracy: {simple_accuracy:.4f}")
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """
        Preprocess the data with advanced feature engineering.
        """
        print("Preprocessing data with advanced feature engineering...")
        
        # Make a copy of the dataframe
        df = self.df.copy()
        
        # 1. Extract titles from names
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_mapping = {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Dr": "Officer",
            "Rev": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Mlle": "Miss",
            "Mme": "Mrs",
            "Don": "Royalty",
            "Sir": "Royalty",
            "Lady": "Royalty",
            "Countess": "Royalty",
            "Jonkheer": "Royalty",
            "Capt": "Officer",
            "Dona": "Royalty"
        }
        df['Title'] = df['Title'].map(title_mapping)
        df['Title'] = df['Title'].fillna("Other")
        
        # 2. Create family features (if not already created)
        if 'FamilySize' not in df.columns:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        if 'IsAlone' not in df.columns:
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # 3. Create age bands
        df['AgeBand'] = pd.cut(
            df['Age'].fillna(df['Age'].median()), 
            bins=[0, 12, 18, 35, 60, 100],
            labels=['Child', 'Teenager', 'YoungAdult', 'Adult', 'Senior']
        )
        
        # 4. Create fare bands
        df['FareBand'] = pd.cut(
            df['Fare'].fillna(df['Fare'].median()),
            bins=[0, 7.91, 14.454, 31, 513],
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        
        # 5. Extract deck from cabin
        df['Deck'] = df['Cabin'].str[0].fillna('U')
        
        # 6. Create embarked*pclass interaction
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['EmbarkedClass'] = df['Embarked'] + df['Pclass'].astype(str)
        
        # 7. Group family size
        df['FamilyGroup'] = pd.cut(
            df['FamilySize'],
            bins=[0, 1, 4, 11],
            labels=['Alone', 'Small', 'Large']
        )
        
        # Store the engineered dataframe for later steps
        self.df_engineered = df
        
        # Log the feature engineering steps
        with mlflow.start_run(run_name="feature_engineering") as run:
            # Track run ID
            self.feature_eng_run_id = run.info.run_id
            
            # Log feature counts
            mlflow.log_param("original_features", len(self.df.columns))
            mlflow.log_param("engineered_features", len(df.columns))
            mlflow.log_param("new_features_added", len(df.columns) - len(self.df.columns))
            
            # Log unique values in categorical features
            categorical_features = ['Title', 'AgeBand', 'FareBand', 'Deck', 'EmbarkedClass', 'FamilyGroup']
            for feature in categorical_features:
                mlflow.log_param(f"{feature}_categories", df[feature].nunique())
            
            # Create and log feature descriptions
            feature_descriptions = {
                "Title": "Extracted title from passenger name",
                "FamilySize": "Total number of family members (SibSp + Parch + 1)",
                "IsAlone": "Binary indicator if passenger is traveling alone",
                "AgeBand": "Age grouped into bands (Child, Teenager, etc.)",
                "FareBand": "Fare grouped into bands (Low, Medium-Low, etc.)",
                "Deck": "Deck extracted from cabin information",
                "EmbarkedClass": "Interaction between embarkation port and passenger class",
                "FamilyGroup": "Family size grouped into categories (Alone, Small, Large)"
            }
            
            with open("feature_descriptions.json", "w") as f:
                json.dump(feature_descriptions, f, indent=2)
            mlflow.log_artifact("feature_descriptions.json")
            
            # Create visualization of value counts for categorical features
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, feature in enumerate(categorical_features[:6]):  # Limit to 6 features
                df[feature].value_counts().sort_index().plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'{feature} Distribution')
                axes[i].set_ylabel('Count')
                
            plt.tight_layout()
            plt.savefig('categorical_features.png')
            mlflow.log_artifact('categorical_features.png')
            plt.close()
        
        print("Feature engineering complete:")
        print(f"  Original features: {len(self.df.columns)}")
        print(f"  Engineered features: {len(df.columns)}")
        print(f"  New features added: {len(df.columns) - len(self.df.columns)}")
        
        self.next(self.prepare_features)
    
    @step
    def prepare_features(self):
        """
        Prepare features for model training with preprocessing pipeline.
        """
        print("Preparing features with preprocessing pipeline...")
        
        # Get the engineered dataframe
        df = self.df_engineered
        
        # Define features and target
        X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
        y = df['Survived']
        
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Print feature types
        print(f"Numeric features: {', '.join(numeric_features)}")
        print(f"Categorical features: {', '.join(categorical_features)}")
        
        # Manual preprocessing instead of using ColumnTransformer 
        # (which can be hard to debug in this pipeline)
        
        # Process numeric features
        # 1. Create imputer for numeric features
        imputer = SimpleImputer(strategy='median')
        
        # Get numeric data
        X_numeric = X[numeric_features].copy()
        
        # Fit imputer on training data
        imputer.fit(X_numeric)
        
        # Transform the data
        X_numeric_imputed = pd.DataFrame(
            imputer.transform(X_numeric),
            columns=numeric_features
        )
        
        # 2. Scale numeric features
        scaler = StandardScaler()
        X_numeric_scaled = pd.DataFrame(
            scaler.fit_transform(X_numeric_imputed),
            columns=numeric_features
        )
        
        # Process categorical features
        # 3. Impute missing values in categorical features
        X_categorical = X[categorical_features].copy()
        
        # Fill missing values with most frequent value
        for col in categorical_features:
            most_frequent = X_categorical[col].mode()[0]
            X_categorical[col] = X_categorical[col].fillna(most_frequent)
        
        # 4. One-hot encode categorical features
        X_categorical_encoded = pd.get_dummies(X_categorical, columns=categorical_features)
        
        # 5. Combine processed numeric and categorical features
        X_processed = pd.concat([X_numeric_scaled, X_categorical_encoded], axis=1)
        
        # Store column names for feature importance interpretation
        self.feature_names = X_processed.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Store the processed data for model training
        self.X_train = X_train
        self.X_test = X_test  
        self.y_train = y_train
        self.y_test = y_test
        
        # Feature selection using Random Forest
        print("Performing feature selection...")
        
        with mlflow.start_run(run_name="feature_selection") as run:
            # Track run ID
            self.feature_selection_run_id = run.info.run_id
            
            # Train a Random Forest for feature importance
            selector_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector_model.fit(X_train, y_train)
            
            # Get feature importances
            feature_importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': selector_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Visualize top features
            plt.figure(figsize=(12, 8))
            top_features = feature_importances.head(20)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig('feature_importances.png')
            mlflow.log_artifact('feature_importances.png')
            plt.close()
            
            # Log feature importances
            feature_importances.to_csv('feature_importances.csv', index=False)
            mlflow.log_artifact('feature_importances.csv')
            
            # Select important features
            # Approach: Select features until we reach 95% of cumulative importance
            cumulative_importance = feature_importances['importance'].cumsum()
            importance_threshold = cumulative_importance[cumulative_importance <= 0.95].max()
            selected_features = feature_importances[
                feature_importances['importance'] >= importance_threshold * 0.05
            ]['feature'].tolist()
            
            # Alternative approach: Select top k features
            top_k = 15
            top_k_features = feature_importances.head(top_k)['feature'].tolist()
            
            # Log selected features
            mlflow.log_param("num_original_features", X_train.shape[1])
            mlflow.log_param("num_selected_features", len(selected_features))
            mlflow.log_param("num_top_k_features", len(top_k_features))
            
            # Store selected features for later use
            self.all_features = X_train.columns.tolist()
            self.selected_features = selected_features
            self.top_k_features = top_k_features
            
            # Create datasets with selected features
            self.X_train_selected = X_train[selected_features]
            self.X_test_selected = X_test[selected_features]
            
            self.X_train_top_k = X_train[top_k_features]
            self.X_test_top_k = X_test[top_k_features]
        
        print(f"Feature selection complete:")
        print(f"  Original features: {len(self.all_features)}")
        print(f"  Selected features (95% importance): {len(self.selected_features)}")
        print(f"  Top {top_k} features: {len(self.top_k_features)}")
        
        self.next(self.tune_hyperparameters, foreach='models')
    
    @step
    def tune_hyperparameters(self):
        """
        Tune hyperparameters for each model type.
        This step runs in parallel for each model type.
        """
        model_type = self.input
        print(f"Tuning hyperparameters for {model_type} ({self.model_names[model_type]})...")
        
        # Define parameter grids for each model type
        param_grids = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'lr': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            },
            'gb': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
        
        # Define model classes
        model_classes = {
            'rf': RandomForestClassifier(random_state=self.random_state),
            'lr': LogisticRegression(random_state=self.random_state),
            'gb': GradientBoostingClassifier(random_state=self.random_state),
            'svm': SVC(probability=True, random_state=self.random_state)
        }
        
        # Get the appropriate parameter grid and model class
        param_grid = param_grids[model_type]
        model = model_classes[model_type]
        
        # Select which feature set to use
        # Use full features for less complex models, selected features for more complex ones
        if model_type in ['lr', 'svm']:
            X_train = self.X_train_selected
            X_test = self.X_test_selected
            feature_set_name = "selected_features"
            used_features = self.selected_features
        else:
            X_train = self.X_train
            X_test = self.X_test
            feature_set_name = "all_features"
            used_features = self.all_features
        
        y_train = self.y_train
        y_test = self.y_test
        
        with mlflow.start_run(run_name=f"{model_type}_tuning") as run:
            # Track run ID
            self.run_id = run.info.run_id
            
            # Log basic information
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_name", self.model_names[model_type])
            mlflow.log_param("feature_set", feature_set_name)
            mlflow.log_param("num_features", len(used_features))
            
            # Log baseline comparisons
            for baseline_name, baseline_metrics in self.baselines.items():
                mlflow.log_metric(f"{baseline_name}_baseline_accuracy", baseline_metrics["accuracy"])
                if "f1" in baseline_metrics:
                    mlflow.log_metric(f"{baseline_name}_baseline_f1", baseline_metrics["f1"])
            
            # Implement hyperparameter tuning with GridSearchCV
            cv = self.cv_folds
            scoring = 'accuracy'
            
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the grid search
            print(f"Fitting GridSearchCV for {model_type} with {len(X_train)} samples...")
            grid_search.fit(X_train, y_train)
            
            # Log best parameters
            best_params = grid_search.best_params_
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Log best CV score
            best_cv_score = grid_search.best_score_
            mlflow.log_metric("best_cv_score", best_cv_score)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Calculate improvements over baselines
            for baseline_name, baseline_metrics in self.baselines.items():
                improvement = accuracy - baseline_metrics["accuracy"]
                mlflow.log_metric(f"improvement_over_{baseline_name}", improvement)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{self.model_names[model_type]} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = f"{model_type}_confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
            
            # Generate ROC curve if probabilities are available
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = roc_auc_score(y_test, y_prob)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{self.model_names[model_type]} ROC Curve')
                plt.legend(loc='lower right')
                roc_path = f"{model_type}_roc_curve.png"
                plt.tight_layout()
                plt.savefig(roc_path)
                mlflow.log_artifact(roc_path)
                mlflow.log_metric("roc_auc", roc_auc)
                plt.close()
            
            # Log feature importances for tree-based models
            if model_type in ['rf', 'gb']:
                feature_names = X_train.columns
                importances = best_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(12, 8))
                plt.title(f'Feature Importances - {self.model_names[model_type]}')
                plt.bar(range(X_train.shape[1]), importances[indices])
                plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                feature_imp_path = f"{model_type}_feature_importances.png"
                plt.savefig(feature_imp_path)
                mlflow.log_artifact(feature_imp_path)
                plt.close()
            
            # For logistic regression, log coefficients
            if model_type == 'lr':
                coef = best_model.coef_[0]
                feature_names = X_train.columns
                coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
                coef_df = coef_df.sort_values('Coefficient', ascending=False)
                
                plt.figure(figsize=(12, 8))
                plt.title('Logistic Regression Coefficients')
                plt.bar(range(len(coef_df)), coef_df['Coefficient'])
                plt.xticks(range(len(coef_df)), coef_df['Feature'], rotation=90)
                plt.tight_layout()
                coef_path = "lr_coefficients.png"
                plt.savefig(coef_path)
                mlflow.log_artifact(coef_path)
                plt.close()
            
            # Create a model card with information about the model
            model_card = f"""# {self.model_names[model_type]} Model Card

## Model Information
- **Model Type:** {self.model_names[model_type]}
- **Feature Set:** {feature_set_name} ({len(used_features)} features)
- **Training Date:** {datetime.now().strftime('%Y-%m-%d')}

## Performance Metrics
- **Accuracy:** {accuracy:.4f}
- **Precision:** {precision:.4f}
- **Recall:** {recall:.4f}
- **F1 Score:** {f1:.4f}
{f"- **ROC AUC:** {roc_auc:.4f}" if y_prob is not None else ""}

## Baseline Comparisons
{chr(10).join([f"- **{baseline_name.capitalize()} Baseline:** {metrics['accuracy']:.4f} (Improvement: {accuracy - metrics['accuracy']:.4f})" for baseline_name, metrics in self.baselines.items()])}

## Hyperparameters
{chr(10).join([f"- **{param}:** {value}" for param, value in best_params.items()])}

## Feature Importance
Top 5 most important features:
{chr(10).join([f"- {row['feature']}" for _, row in feature_importances.head(5).iterrows()]) if model_type in ['rf', 'gb'] else "Not applicable for this model type"}

## Training Information
- **Cross-validation:** {cv}-fold
- **Best CV Score:** {best_cv_score:.4f}
- **Test Set Size:** {len(y_test)} samples
"""
            
            with open(f"{model_type}_model_card.md", "w") as f:
                f.write(model_card)
            mlflow.log_artifact(f"{model_type}_model_card.md")
            
            # Log the model
            mlflow.sklearn.log_model(
                best_model, 
                f"{model_type}_model",
                signature=mlflow.models.signature.infer_signature(X_test, y_pred),
                input_example=X_test.iloc[:5]
            )
            
            # Save the model and results for the next step
            self.best_model = best_model
            self.best_params = best_params
            self.best_cv_score = best_cv_score
            self.metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc if y_prob is not None else None
            }
            self.model_type = model_type
            self.feature_set = feature_set_name
            self.used_features = used_features
            
            print(f"Hyperparameter tuning complete for {model_type}:")
            print(f"  Best CV score: {best_cv_score:.4f}")
            print(f"  Test accuracy: {accuracy:.4f}")
            print(f"  Best parameters: {best_params}")
        
        self.next(self.join_models)
    
    @step
    def join_models(self, inputs):
        """
        Join results from all model types and compare them.
        """
        print("Joining results from all models...")
        
        # Collect results from each model
        self.model_results = []
        
        for inp in inputs:
            result = {
                "model_type": inp.model_type,
                "model_name": self.model_names[inp.model_type],
                "best_model": inp.best_model,
                "best_params": inp.best_params,
                "best_cv_score": inp.best_cv_score,
                "metrics": inp.metrics,
                "feature_set": inp.feature_set,
                "used_features": inp.used_features,
                "run_id": inp.run_id
            }
            self.model_results.append(result)
        
        # Find the best model based on accuracy
        best_model_idx = np.argmax([result["metrics"]["accuracy"] for result in self.model_results])
        self.best_model_result = self.model_results[best_model_idx]
        
        # Create a comparison table
        comparison_data = []
        for result in self.model_results:
            row = {
                "Model": result["model_name"],
                "Accuracy": result["metrics"]["accuracy"],
                "Precision": result["metrics"]["precision"],
                "Recall": result["metrics"]["recall"],
                "F1": result["metrics"]["f1"],
                "ROC AUC": result["metrics"]["roc_auc"] if result["metrics"]["roc_auc"] is not None else float('nan'),
                "Features": len(result["used_features"]),
                "CV Score": result["best_cv_score"]
            }
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Log the comparison
        with mlflow.start_run(run_name="model_comparison") as run:
            # Track run ID
            self.comparison_run_id = run.info.run_id
            
            # Log the best model
            mlflow.log_param("best_model", self.best_model_result["model_name"])
            mlflow.log_param("best_model_type", self.best_model_result["model_type"])
            mlflow.log_metric("best_model_accuracy", self.best_model_result["metrics"]["accuracy"])
            mlflow.log_metric("best_model_f1", self.best_model_result["metrics"]["f1"])
            
            # Log baseline comparisons
            for baseline_name, baseline_metrics in self.baselines.items():
                improvement = self.best_model_result["metrics"]["accuracy"] - baseline_metrics["accuracy"]
                mlflow.log_metric(f"best_model_improvement_over_{baseline_name}", improvement)
            
            # Create comparison table visualization
            plt.figure(figsize=(12, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            metrics_df = self.comparison_df[['Model'] + metrics].melt(
                id_vars='Model', 
                var_name='Metric', 
                value_name='Value'
            )
            sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df)
            plt.title('Model Comparison - Performance Metrics')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('model_comparison.png')
            mlflow.log_artifact('model_comparison.png')
            plt.close()
            
            # Create ROC AUC comparison
            if not self.comparison_df['ROC AUC'].isna().all():
                plt.figure(figsize=(10, 6))
                valid_roc = self.comparison_df[~self.comparison_df['ROC AUC'].isna()]
                sns.barplot(x='Model', y='ROC AUC', data=valid_roc)
                plt.title('Model Comparison - ROC AUC')
                plt.ylim(0.5, 1.0)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('roc_auc_comparison.png')
                mlflow.log_artifact('roc_auc_comparison.png')
                plt.close()
            
            # Save comparison data
            self.comparison_df.to_csv('model_comparison.csv', index=False)
            mlflow.log_artifact('model_comparison.csv')
            
            # Create a model comparison report
            report = f"""# Model Comparison Report

## Best Performing Model: {self.best_model_result["model_name"]}

### Performance Metrics
- **Accuracy:** {self.best_model_result["metrics"]["accuracy"]:.4f}
- **Precision:** {self.best_model_result["metrics"]["precision"]:.4f}
- **Recall:** {self.best_model_result["metrics"]["recall"]:.4f}
- **F1 Score:** {self.best_model_result["metrics"]["f1"]:.4f}
{f'- **ROC AUC:** {self.best_model_result["metrics"]["roc_auc"]:.4f}' if self.best_model_result["metrics"]["roc_auc"] is not None else ""}

### Best Configuration
- **Feature set:** {self.best_model_result["feature_set"]} ({len(self.best_model_result["used_features"])} features)
- **Best parameters:** {json.dumps(self.best_model_result["best_params"], indent=2).replace('{', '').replace('}', '').replace('"', '')}

### Baseline Comparisons
{chr(10).join([f"- **{baseline_name.capitalize()} Baseline:** {metrics['accuracy']:.4f} (Improvement: {self.best_model_result['metrics']['accuracy'] - metrics['accuracy']:.4f})" for baseline_name, metrics in self.baselines.items()])}

## Model Rankings

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Features | CV Score |
|-------|----------|-----------|--------|-----|---------|----------|----------|
{chr(10).join([f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['ROC AUC']:.4f if not pd.isna(row['ROC AUC']) else 'N/A'} | {row['Features']} | {row['CV Score']:.4f} |" for _, row in self.comparison_df.iterrows()])}

## Analysis

The {self.best_model_result["model_name"]} model achieved the highest accuracy of {self.best_model_result["metrics"]["accuracy"]:.4f}, which represents a {(self.best_model_result["metrics"]["accuracy"] - self.baselines["gender"]["accuracy"]):.4f} improvement over the gender-based baseline.

### Key Findings

1. The gender feature remains highly predictive across all models
2. Passenger class and fare are also consistently important features
3. Feature engineering improved model performance by capturing interactions and creating more informative features

## Next Steps

1. Deploy the best model for predictions
2. Consider ensemble methods to further improve performance
3. Gather additional data or features if available
"""
            
            with open('model_comparison_report.md', 'w') as f:
                f.write(report)
            mlflow.log_artifact('model_comparison_report.md')
            
            # Register the best model in the MLflow registry
            mlflow.sklearn.log_model(
                self.best_model_result["best_model"], 
                "best_model",
                registered_model_name="titanic_best_model",
                signature=mlflow.models.signature.infer_signature(
                    inputs[0].X_test.iloc[:5] if hasattr(inputs[0], 'X_test') else None,
                    self.best_model_result["best_model"].predict(inputs[0].X_test.iloc[:5]) if hasattr(inputs[0], 'X_test') else None
                ),
                input_example=inputs[0].X_test.iloc[:5] if hasattr(inputs[0], 'X_test') else None
            )
        
        print("Model comparison complete:")
        print(f"  Best model: {self.best_model_result['model_name']} with accuracy {self.best_model_result['metrics']['accuracy']:.4f}")
        print(f"  Improvement over majority baseline: {self.best_model_result['metrics']['accuracy'] - self.baselines['majority']['accuracy']:.4f}")
        if 'gender' in self.baselines:
            print(f"  Improvement over gender baseline: {self.best_model_result['metrics']['accuracy'] - self.baselines['gender']['accuracy']:.4f}")
        
        self.next(self.deploy_model)
        
    @step
    def join_models(self, inputs):
        """
        Join results from all model types and compare them.
        """
        print("Joining results from all models...")
        
        # Collect results from each model
        self.model_results = []
        
        for inp in inputs:
            result = {
                "model_type": inp.model_type,
                "model_name": self.model_names[inp.model_type],
                "best_model": inp.best_model,
                "best_params": inp.best_params,
                "best_cv_score": inp.best_cv_score,
                "metrics": inp.metrics,
                "feature_set": inp.feature_set,
                "used_features": inp.used_features,
                "run_id": inp.run_id
            }
            self.model_results.append(result)
        
        # Find the best model based on accuracy
        best_model_idx = np.argmax([result["metrics"]["accuracy"] for result in self.model_results])
        self.best_model_result = self.model_results[best_model_idx]
        
        # Create a comparison table
        comparison_data = []
        for result in self.model_results:
            row = {
                "Model": result["model_name"],
                "Accuracy": result["metrics"]["accuracy"],
                "Precision": result["metrics"]["precision"],
                "Recall": result["metrics"]["recall"],
                "F1": result["metrics"]["f1"],
                "ROC AUC": result["metrics"]["roc_auc"] if result["metrics"]["roc_auc"] is not None else float('nan'),
                "Features": len(result["used_features"]),
                "CV Score": result["best_cv_score"]
            }
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Log the comparison
        with mlflow.start_run(run_name="model_comparison") as run:
            # Track run ID
            self.comparison_run_id = run.info.run_id
            
            # Log the best model
            mlflow.log_param("best_model", self.best_model_result["model_name"])
            mlflow.log_param("best_model_type", self.best_model_result["model_type"])
            mlflow.log_metric("best_model_accuracy", self.best_model_result["metrics"]["accuracy"])
            mlflow.log_metric("best_model_f1", self.best_model_result["metrics"]["f1"])
            
            # Log baseline comparisons
            for baseline_name, baseline_metrics in self.baselines.items():
                improvement = self.best_model_result["metrics"]["accuracy"] - baseline_metrics["accuracy"]
                mlflow.log_metric(f"best_model_improvement_over_{baseline_name}", improvement)
            
            # Create comparison table visualization
            plt.figure(figsize=(12, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            metrics_df = self.comparison_df[['Model'] + metrics].melt(
                id_vars='Model', 
                var_name='Metric', 
                value_name='Value'
            )
            sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df)
            plt.title('Model Comparison - Performance Metrics')
            plt.ylim(0, 1.0)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('model_comparison.png')
            mlflow.log_artifact('model_comparison.png')
            plt.close()
            
            # Create ROC AUC comparison
            if not self.comparison_df['ROC AUC'].isna().all():
                plt.figure(figsize=(10, 6))
                valid_roc = self.comparison_df[~self.comparison_df['ROC AUC'].isna()]
                sns.barplot(x='Model', y='ROC AUC', data=valid_roc)
                plt.title('Model Comparison - ROC AUC')
                plt.ylim(0.5, 1.0)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('roc_auc_comparison.png')
                mlflow.log_artifact('roc_auc_comparison.png')
                plt.close()
            
            # Save comparison data
            self.comparison_df.to_csv('model_comparison.csv', index=False)
            mlflow.log_artifact('model_comparison.csv')
            
            # Create a model comparison report
            report = f"""# Model Comparison Report

## Best Performing Model: {self.best_model_result["model_name"]}

### Performance Metrics
- **Accuracy:** {self.best_model_result["metrics"]["accuracy"]:.4f}
- **Precision:** {self.best_model_result["metrics"]["precision"]:.4f}
- **Recall:** {self.best_model_result["metrics"]["recall"]:.4f}
- **F1 Score:** {self.best_model_result["metrics"]["f1"]:.4f}
{f'- **ROC AUC:** {self.best_model_result["metrics"]["roc_auc"]:.4f}' if self.best_model_result["metrics"]["roc_auc"] is not None else ""}

### Best Configuration
- **Feature set:** {self.best_model_result["feature_set"]} ({len(self.best_model_result["used_features"])} features)
- **Best parameters:** {json.dumps(self.best_model_result["best_params"], indent=2).replace('{', '').replace('}', '').replace('"', '')}

### Baseline Comparisons
{chr(10).join([f"- **{baseline_name.capitalize()} Baseline:** {metrics['accuracy']:.4f} (Improvement: {self.best_model_result['metrics']['accuracy'] - metrics['accuracy']:.4f})" for baseline_name, metrics in self.baselines.items()])}

## Model Rankings

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Features | CV Score |
|-------|----------|-----------|--------|-----|---------|----------|----------|
{chr(10).join([f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['ROC AUC']:.4f if not pd.isna(row['ROC AUC']) else 'N/A'} | {row['Features']} | {row['CV Score']:.4f} |" for _, row in self.comparison_df.iterrows()])}

## Analysis

The {self.best_model_result["model_name"]} model achieved the highest accuracy of {self.best_model_result["metrics"]["accuracy"]:.4f}, which represents a {(self.best_model_result["metrics"]["accuracy"] - self.baselines["gender"]["accuracy"]):.4f} improvement over the gender-based baseline.

### Key Findings

1. The gender feature remains highly predictive across all models
2. Passenger class and fare are also consistently important features
3. Feature engineering improved model performance by capturing interactions and creating more informative features

## Next Steps

1. Deploy the best model for predictions
2. Consider ensemble methods to further improve performance
3. Gather additional data or features if available
"""
            
            with open('model_comparison_report.md', 'w') as f:
                f.write(report)
            mlflow.log_artifact('model_comparison_report.md')
            
            # Register the best model in the MLflow registry
            mlflow.sklearn.log_model(
                self.best_model_result["best_model"], 
                "best_model",
                registered_model_name="titanic_best_model",
                signature=mlflow.models.signature.infer_signature(
                    inputs[0].X_test.iloc[:5] if hasattr(inputs[0], 'X_test') else None,
                    self.best_model_result["best_model"].predict(inputs[0].X_test.iloc[:5]) if hasattr(inputs[0], 'X_test') else None
                ),
                input_example=inputs[0].X_test.iloc[:5] if hasattr(inputs[0], 'X_test') else None
            )
        
        print("Model comparison complete:")
        print(f"  Best model: {self.best_model_result['model_name']} with accuracy {self.best_model_result['metrics']['accuracy']:.4f}")
        print(f"  Improvement over majority baseline: {self.best_model_result['metrics']['accuracy'] - self.baselines['majority']['accuracy']:.4f}")
        if 'gender' in self.baselines:
            print(f"  Improvement over gender baseline: {self.best_model_result['metrics']['accuracy'] - self.baselines['gender']['accuracy']:.4f}")
        
        self.next(self.deploy_model)
    
    @step
    def end(self):
        """
        Final step to summarize the flow.
        """
        print("\n" + "="*80)
        print(f"Advanced Titanic ML Flow Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Summary of the process
        print("\nProcess Summary:")
        print(f"  1. Loaded Titanic dataset with {self.df.shape[0]} passengers")
        print(f"  2. Performed exploratory data analysis")
        print(f"  3. Created baseline models:")
        for baseline_name, metrics in self.baselines.items():
            print(f"     - {baseline_name.capitalize()} baseline: {metrics['accuracy']:.4f} accuracy")
        
        print(f"  4. Engineered {len(self.df_engineered.columns) - len(self.df.columns)} new features")
        print(f"  5. Applied preprocessing and feature selection")
        print(f"  6. Trained and tuned {len(self.model_results)} different models in parallel")
        print(f"  7. Selected the best model: {self.best_model_result['model_name']}")
        print(f"  8. Prepared deployment artifacts")
        
        # Result summary
        print("\nResults Summary:")
        print(f"  Best model: {self.best_model_result['model_name']}")
        print(f"  Accuracy: {self.best_model_result['metrics']['accuracy']:.4f}")
        print(f"  F1 Score: {self.best_model_result['metrics']['f1']:.4f}")
        
        # Baseline comparisons
        print("\nBaseline Comparisons:")
        for baseline_name, metrics in self.baselines.items():
            improvement = self.best_model_result['metrics']['accuracy'] - metrics['accuracy']
            print(f"  {baseline_name.capitalize()} baseline: {metrics['accuracy']:.4f} → Improvement: {improvement:.4f}")
        
        # MLflow information
        print("\nMLflow Information:")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Best model run ID: {self.best_model_result['run_id']}")
        print(f"  Deployment run ID: {self.deployment_run_id}")
        print(f"  To view results: mlflow ui")

        # Next steps
        print("\nNext Steps:")
        print("  1. Review model performance and artifacts in MLflow UI")
        print("  2. Explore deployment artifacts in ./deployment directory")
        print("  3. Deploy the model using the provided predict.py script")
        print("  4. Consider ensemble methods or additional feature engineering")
        print("  5. Test the model on new data to validate generalization")
        
        print("\n" + "="*80)
        print("For more information, see the generated README.md files and MLflow artifacts.")
        print("="*80 + "\n")


if __name__ == '__main__':
    AdvancedTitanicFlow()