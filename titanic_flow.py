from metaflow import FlowSpec, step, Parameter, IncludeFile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

class TitanicFlow(FlowSpec):
    """
    A flow to train a machine learning model on the Titanic dataset
    and track experiments with MLflow.
    """
    
    # Define parameters
    data_path = Parameter('data_path', 
                         help='Path to the Titanic dataset',
                         default='./data/train.csv')
    
    test_size = Parameter('test_size',
                         help='Fraction of data to use for testing',
                         default=0.2)
    
    n_estimators = Parameter('n_estimators',
                            help='Number of trees in the forest',
                            default=100)
    
    random_state = Parameter('random_state',
                           help='Random seed',
                           default=42)
    
    @step
    def start(self):
        """
        Start the flow and check if the data file exists.
        If not, download it from Kaggle.
        """
        import os
        
        if not os.path.exists(self.data_path):
            print(f"Data file not found at {self.data_path}.")
            print("Please download the Titanic dataset from Kaggle and place it in the data folder.")
            print("Or you can use the following commands:")
            print("mkdir -p data")
            print("kaggle competitions download -c titanic -p ./data")
            print("unzip ./data/titanic.zip -d ./data")
            raise FileNotFoundError(f"Data file not found at {self.data_path}.")
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """
        Load the Titanic dataset and perform basic data exploration.
        """
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        print("Dataset shape:", self.df.shape)
        print("\nColumn names:", self.df.columns.tolist())
        print("\nData types:\n", self.df.dtypes)
        print("\nMissing values:\n", self.df.isnull().sum())
        
        self.next(self.preprocess_data)
    
    @step
    def preprocess_data(self):
        """
        Preprocess the data for machine learning.
        """
        print("Preprocessing data...")
        
        # Make a copy of the dataframe
        df = self.df.copy()
        
        # Feature engineering
        
        # 1. Handle missing values
        # Fill missing age with median
        df['Age'] = df['Age'].fillna(df['Age'].median())
        # Fill missing embarked with mode
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        # Drop Cabin as it has too many missing values
        df = df.drop(['Cabin'], axis=1)
        
        # 2. Extract title from name
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # Group rare titles
        rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
        df.loc[df['Title'].isin(rare_titles), 'Title'] = 'Rare'
        df.loc[df['Title'] == 'Mlle', 'Title'] = 'Miss'
        df.loc[df['Title'] == 'Ms', 'Title'] = 'Miss'
        df.loc[df['Title'] == 'Mme', 'Title'] = 'Mrs'
        
        # 3. Convert categorical features to numeric
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping)
        df['Title'] = df['Title'].fillna(0)
        
        # 4. Create family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # 5. Create is_alone feature
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
        
        # 6. Drop unnecessary columns
        df = df.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
        
        print("Processed dataframe shape:", df.shape)
        print("Processed columns:", df.columns.tolist())
        
        # Store the processed dataframe
        self.processed_df = df
        
        self.next(self.prepare_features)
    
    @step
    def prepare_features(self):
        """
        Prepare features and target for machine learning.
        """
        print("Preparing features and target...")
        
        # Define features and target
        X = self.processed_df.drop(['Survived'], axis=1)
        y = self.processed_df['Survived']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = X.columns.tolist()
        
        print(f"Train set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        self.next(self.create_baselines)

    @step
    def create_baselines(self):
        """
        Create baseline models for comparison.
        """
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score
        
        print("Creating baseline models...")
        
        # 1. Majority class baseline
        majority_clf = DummyClassifier(strategy='most_frequent')
        majority_clf.fit(self.X_train, self.y_train)
        majority_pred = majority_clf.predict(self.X_test)
        majority_accuracy = accuracy_score(self.y_test, majority_pred)
        
        # 2. Gender-based baseline (custom implementation)
        gender_accuracy = 0
        
        # Check if 'Sex' column exists in the DataFrame
        if 'Sex' in self.X_test.columns:
            # Direct column access for pandas DataFrame
            gender_pred = (self.X_test['Sex'] == 1).astype(int)
            gender_accuracy = accuracy_score(self.y_test, gender_pred)
            print(f"Created gender-based baseline using 'Sex' column")
        else:
            # Try to find a column that might be the Sex feature
            sex_related_cols = [col for col in self.X_test.columns if 'sex' in col.lower()]
            
            if sex_related_cols:
                # Try the first column that might be sex-related
                col = sex_related_cols[0]
                gender_pred = (self.X_test[col] == 1).astype(int)
                gender_accuracy = accuracy_score(self.y_test, gender_pred)
                print(f"Created gender-based baseline using '{col}' column")
            else:
                print("Cannot create gender baseline: No sex-related column found")
        
        # Log baselines to MLflow
        with mlflow.start_run(run_name="baselines") as run:
            mlflow.log_metric("majority_baseline_accuracy", majority_accuracy)
            if gender_accuracy > 0:
                mlflow.log_metric("gender_baseline_accuracy", gender_accuracy)
        
        # Store for later comparison
        self.baselines = {
            "majority": majority_accuracy
        }
        
        if gender_accuracy > 0:
            self.baselines["gender"] = gender_accuracy
        
        print(f"Baseline accuracies:")
        print(f"  Majority class: {majority_accuracy:.4f}")
        if gender_accuracy > 0:
            print(f"  Gender-based: {gender_accuracy:.4f}")
        
        self.next(self.train_model)
    
    @step
    def train_model(self):
        """
        Train a RandomForest model and track with MLflow.
        """
        print("Training RandomForest model...")
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("titanic-survival-prediction")
        
        # Start an MLflow run
        with mlflow.start_run(run_name="rf_model") as run:
            run_id = run.info.run_id
            print(f"MLflow Run ID: {run_id}")
            
            # Log parameters
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("random_state", self.random_state)
            
            # Log baseline metrics for comparison
            for baseline_name, baseline_value in self.baselines.items():
                mlflow.log_metric(f"{baseline_name}_baseline", baseline_value)
            
            # Train the model
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Calculate improvements over baselines
            improvement_over_majority = accuracy - self.baselines["majority"]
            improvement_over_gender = accuracy - self.baselines.get("gender", 0)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("improvement_over_majority", improvement_over_majority)
            if "gender" in self.baselines and self.baselines["gender"] > 0:
                mlflow.log_metric("improvement_over_gender", improvement_over_gender)
            
            # Log feature importances
            feature_importances = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance_path = "feature_importances.csv"
            feature_importances.to_csv(feature_importance_path, index=False)
            mlflow.log_artifact(feature_importance_path)
            
            # Create feature importance visualization
            plt.figure(figsize=(10, 6))
            top_features = feature_importances.head(10)
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            feature_plot_path = "feature_importance_plot.png"
            plt.savefig(feature_plot_path)
            mlflow.log_artifact(feature_plot_path)
            
            # Create confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_plot_path = "confusion_matrix.png"
            plt.savefig(cm_plot_path)
            mlflow.log_artifact(cm_plot_path)
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
            
            # Store metrics and model for next steps
            self.metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "improvement_over_majority": improvement_over_majority,
                "improvement_over_gender": improvement_over_gender if "gender" in self.baselines else 0
            }
            self.model = model
            self.run_id = run_id
            
            print(f"Model training complete!")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Baseline comparisons:")
            print(f"  Majority baseline: {self.baselines['majority']:.4f}")
            print(f"  Improvement over majority: {improvement_over_majority:.4f}")
            if "gender" in self.baselines and self.baselines["gender"] > 0:
                print(f"  Gender baseline: {self.baselines['gender']:.4f}")
                print(f"  Improvement over gender: {improvement_over_gender:.4f}")
            
            # Save the model as a pickle file
            model_path = "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        End the flow and print summary.
        """
        print("\n" + "="*50)
        print("Titanic ML Flow Complete!")
        print("="*50)
        print(f"Model metrics:")
        for metric, value in self.metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nBaseline comparisons:")
        print(f"  Majority baseline: {self.baselines['majority']:.4f}")
        if "gender" in self.baselines and self.baselines["gender"] > 0:
            print(f"  Gender baseline: {self.baselines['gender']:.4f}")
            print(f"  Improvement over gender: {self.metrics['accuracy'] - self.baselines['gender']:.4f}")
        print(f"  Improvement over majority: {self.metrics['accuracy'] - self.baselines['majority']:.4f}")
        
        print(f"\nMLflow Run ID: {self.run_id}")
        print(f"Check MLflow UI for detailed results: mlflow ui")
        print("="*50)

if __name__ == '__main__':
    TitanicFlow()