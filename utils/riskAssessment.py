import os
import logging
import json
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('risk_assessment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('risk_assessment')

def load_config(config_path='config.json'):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using default values.")
        return {
            "risk_thresholds": {
                "high_risk": {
                    "pe_ratio": 30,
                    "pb_ratio": 3,
                    "max_volatility": 70
                },
                "moderate_high_risk": {
                    "pe_ratio": 25,
                    "pb_ratio": 2.5,
                    "max_volatility": 60
                },
                "medium_risk": {
                    "pe_ratio": 20,
                    "pb_ratio": 2,
                    "max_volatility": 50
                },
                "moderate_low_risk": {
                    "pe_ratio": 15,
                    "pb_ratio": 1.75,
                    "max_volatility": 40
                },
                "low_risk": {
                    "pe_ratio": 10,
                    "pb_ratio": 1.5,
                    "max_volatility": 30
                }
            },
            "model_params": {
                "random_forest": {
                    "n_estimators": 100,
                    "random_state": 42
                }
            },
            "train_test_split": {
                "test_size": 0.2,
                "random_state": 42
            }
        }

class JSERiskAssessment:
    def __init__(self, config_path='config.json'):
        self.config = load_config(config_path)
        self.create_directories()

    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['data', 'models', 'output']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def extract_data_from_pdf(self, pdf_path, output_csv_path):
        """Extract data from PDF and save to CSV"""
        try:
            logger.info(f"Extracting data from {pdf_path}")
            pdf = pdfplumber.open(pdf_path)
            data = []

            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        data.extend(table)

            if not data:
                logger.error("No tables found in the PDF")
                return None

            columns = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=columns)

            df.to_csv(output_csv_path, index=False)
            logger.info(f"Data extracted and saved to '{output_csv_path}'")
            return df

        except Exception as e:
            logger.error(f"Error extracting data from PDF: {str(e)}")
            return None

    def preprocess_data(self, input_csv_path, output_csv_path):
        """Clean and preprocess the extracted data"""
        try:
            logger.info(f"Preprocessing data from {input_csv_path}")
            df = pd.read_csv(input_csv_path)

            columns_to_drop = ["COMPANY NAME", "SYMBOL"]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            numerical_columns = [
                "MARKET CAPITALIZATION", "CLOSE PRICE", "TRAILING 12M P/E RATIO",
                "P/B RATIO", "52-WEEK HIGH", "52-WEEK LOW", "1-DAY CHANGE (%)",
                "90-DAY CHANGE (%)", "5-YEAR CHANGE (%)", "YTD CHANGE (%)",
                "TRAILING 12-M PROFIT", "TRAILING 12M EPS", "SHAREHOLDERS' EQUITY",
                "BVPS", "TOTAL PREV YR DIV ($)"
            ]

            for col in numerical_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            missing_values = df.isnull().sum()
            logger.info(f"Missing values before imputation:\n{missing_values[missing_values > 0]}")

            df = df.fillna(df.median())

            df.to_csv(output_csv_path, index=False)
            logger.info(f"Data cleaned and saved to '{output_csv_path}'")
            return df

        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None

    def assign_risk_level(self, row):
        """Assign risk level based on financial metrics using config thresholds"""
        try:
            thresholds = self.config["risk_thresholds"]
            volatility = row["52-WEEK HIGH"] - row["52-WEEK LOW"]

            pe_score = row["TRAILING 12M P/E RATIO"]
            pb_score = row["P/B RATIO"]
            volatility_score = volatility
            profitability = row["TRAILING 12M EPS"]
            equity = row["SHAREHOLDERS' EQUITY"]

            if profitability < 0 or equity < 0:
                return "High"

            if (pe_score > thresholds["high_risk"]["pe_ratio"] or
                    pb_score > thresholds["high_risk"]["pb_ratio"] or
                    volatility_score > thresholds["high_risk"]["max_volatility"]):
                return "High"

            if (pe_score > thresholds["moderate_high_risk"]["pe_ratio"] or
                    pb_score > thresholds["moderate_high_risk"]["pb_ratio"] or
                    volatility_score > thresholds["moderate_high_risk"]["max_volatility"]):
                return "Moderate High"

            if (pe_score > thresholds["medium_risk"]["pe_ratio"] or
                    pb_score > thresholds["medium_risk"]["pb_ratio"] or
                    volatility_score > thresholds["medium_risk"]["max_volatility"]):
                return "Medium"

            if (pe_score > thresholds["moderate_low_risk"]["pe_ratio"] or
                    pb_score > thresholds["moderate_low_risk"]["pb_ratio"] or
                    volatility_score > thresholds["moderate_low_risk"]["max_volatility"]):
                return "Moderate Low"

            if (pe_score <= thresholds["low_risk"]["pe_ratio"] and
                    pb_score <= thresholds["low_risk"]["pb_ratio"] and
                    volatility_score <= thresholds["low_risk"]["max_volatility"]):
                return "Low"

            return "Moderate Low"

        except Exception as e:
            logger.warning(f"Error assigning risk level to row: {str(e)}. Defaulting to Medium.")
            return "Medium"

    def create_target_variable(self, input_csv_path, output_csv_path):
        """Create the risk level target variable"""
        try:
            logger.info(f"Creating target variable from {input_csv_path}")
            df = pd.read_csv(input_csv_path)

            df["Risk_Level"] = df.apply(self.assign_risk_level, axis=1)

            risk_distribution = df["Risk_Level"].value_counts()
            logger.info(f"Risk level distribution:\n{risk_distribution}")

            df.to_csv(output_csv_path, index=False)
            logger.info(f"Risk levels assigned and saved to '{output_csv_path}'")
            return df

        except Exception as e:
            logger.error(f"Error creating target variable: {str(e)}")
            return None

    def train_model(self, data_path, model_save_path=None):
        """Train a Random Forest model to predict risk levels with cross-validation"""
        try:
            logger.info(f"Training model on data from {data_path}")
            df = pd.read_csv(data_path)

            label_encoder = LabelEncoder()
            df["Risk_Level_Encoded"] = label_encoder.fit_transform(df["Risk_Level"])

            X = df.drop(columns=["Risk_Level", "Risk_Level_Encoded"])
            y = df["Risk_Level_Encoded"]

            feature_names = X.columns.tolist()

            split_config = self.config["train_test_split"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=split_config["test_size"],
                random_state=split_config["random_state"]
            )

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier())
            ])

            # Define the parameter grid for GridSearchCV
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }

            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            logger.info(f"Best model parameters: {grid_search.best_params_}")

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model Accuracy: {accuracy:.4f}")

            class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
            logger.info(f"Classification Report:\n{class_report}")

            cm = confusion_matrix(y_test, y_pred)

            feature_importance = best_model.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            logger.info(f"Feature Importance:\n{feature_importance_df}")

            if model_save_path:
                model_data = {
                    'model': best_model,
                    'label_encoder': label_encoder,
                    'feature_names': feature_names
                }
                with open(model_save_path, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info(f"Model saved to {model_save_path}")

            return best_model, label_encoder, cm, X_test, y_test, y_pred, feature_importance

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None

    def visualize_results(self, model_results, save_dir='output'):
        """Visualize model results with plots"""
        try:
            if not model_results:
                logger.error("No model results to visualize")
                return

            model, label_encoder, cm, X_test, y_test, y_pred, feature_importance = model_results

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()

            cm_path = os.path.join(save_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            logger.info(f"Confusion matrix saved to {cm_path}")
            plt.close()

            indices = np.argsort(feature_importance)[::-1]
            sorted_features = [X_test.columns[i] for i in indices]
            sorted_importance = feature_importance[indices]

            plt.figure(figsize=(12, 8))
            plt.title('Feature Importance')
            plt.bar(range(len(sorted_importance)), sorted_importance, align='center')
            plt.xticks(range(len(sorted_importance)), sorted_features, rotation=90)
            plt.tight_layout()

            fi_path = os.path.join(save_dir, "feature_importance.png")
            plt.savefig(fi_path)
            logger.info(f"Feature importance plot saved to {fi_path}")
            plt.close()

            risk_counts = np.bincount(y_test)
            plt.figure(figsize=(10, 8))
            plt.pie(risk_counts, labels=label_encoder.classes_, autopct='%1.1f%%', startangle=90)
            plt.title('Risk Level Distribution in Test Data')
            plt.axis('equal')

            dist_path = os.path.join(save_dir, "risk_distribution.png")
            plt.savefig(dist_path)
            logger.info(f"Risk distribution plot saved to {dist_path}")
            plt.close()

        except Exception as e:
            logger.error(f"Error visualizing results: {str(e)}")

    def predict_risk(self, model_path, input_data):
        """Predict risk for new data using a saved model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            model = model_data['model']
            label_encoder = model_data['label_encoder']
            feature_names = model_data['feature_names']

            for feature in feature_names:
                if feature not in input_data:
                    input_data[feature] = 0

            input_df = pd.DataFrame([input_data])[feature_names]

            prediction_encoded = model.predict(input_df)[0]
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]

            proba = model.predict_proba(input_df)[0]
            proba_dict = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(proba)}

            return {
                'risk_level': prediction,
                'confidence': proba_dict
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {'error': str(e)}

    def run_pipeline(self, pdf_path):
        """Run the entire JSE risk assessment pipeline"""
        try:
            raw_csv_path = "data/jse_data_raw.csv"
            cleaned_csv_path = "data/jse_data_cleaned.csv"
            labeled_csv_path = "data/jse_data_with_risk.csv"
            model_save_path = "models/risk_assessment_model.pkl"

            logger.info("Step 1: Extracting data from PDF...")
            df_raw = self.extract_data_from_pdf(pdf_path, raw_csv_path)
            if df_raw is None:
                return False

            logger.info("Step 2: Preprocessing the data...")
            df_cleaned = self.preprocess_data(raw_csv_path, cleaned_csv_path)
            if df_cleaned is None:
                return False

            logger.info("Step 3: Creating target variable...")
            df_labeled = self.create_target_variable(cleaned_csv_path, labeled_csv_path)
            if df_labeled is None:
                return False

            logger.info("Step 4: Training and evaluating model...")
            model_results = self.train_model(labeled_csv_path, model_save_path)
            if model_results is None:
                return False

            logger.info("Step 5: Visualizing results...")
            self.visualize_results(model_results)

            logger.info("Pipeline executed successfully!")
            return True

        except Exception as e:
            logger.error(f"Error running pipeline: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    risk_tool = JSERiskAssessment()
    pdf_path = "data/jse_data_raw.pdf"
    success = risk_tool.run_pipeline(pdf_path)

    if success:
        test_company = {
            "MARKET CAPITALIZATION": 5000000000,
            "CLOSE PRICE": 50.25,
            "TRAILING 12M P/E RATIO": 15.3,
            "P/B RATIO": 1.8,
            "52-WEEK HIGH": 60.75,
            "52-WEEK LOW": 40.25,
            "1-DAY CHANGE (%)": 0.5,
            "90-DAY CHANGE (%)": 5.2,
            "5-YEAR CHANGE (%)": 45.3,
            "YTD CHANGE (%)": 8.7,
            "TRAILING 12-M PROFIT": 350000000,
            "TRAILING 12M EPS": 3.25,
            "SHAREHOLDERS' EQUITY": 2800000000,
            "BVPS": 28.5,
            "TOTAL PREV YR DIV ($)": 1.5
        }

        result = risk_tool.predict_risk("models/risk_assessment_model.pkl", test_company)
        logger.info(f"Prediction for test company: {result}")