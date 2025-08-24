#!/usr/bin/env python3
"""
TANAP Boundary Prediction CLI Tool

This script loads trained models and predicts TANAP boundaries on unseen test inventories.
Supports all model types: logistic regression, random forest, XGBoost, SVM, and neural networks.

Usage:
    python predict_boundaries.py --model logistic_regression --input data/test/unseen\ testsets/1123.csv --output predictions.csv
    python predict_boundaries.py --model all --input data/test/unseen\ testsets/ --output results/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("Warning: TensorFlow not available. Neural network models will be skipped.")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading different types of trained models and their preprocessors"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.available_models = self._discover_models()
        self.label_encoder = self._load_label_encoder()
        
    def _discover_models(self) -> List[str]:
        """Discover available trained models"""
        models = []
        model_patterns = {
            'logistic_regression': 'logistic_regression_model.pkl',
            'random_forest': 'random_forest_model.pkl', 
            'xgboost': 'xgboost_model.pkl',
            'svm': 'svm_model.pkl',
            'neural_network': 'neural_network_model.h5' if HAS_TENSORFLOW else None
        }
        
        for model_name, filename in model_patterns.items():
            if filename and (self.models_dir / filename).exists():
                models.append(model_name)
                
        logger.info(f"Discovered models: {models}")
        return models
    
    def _load_label_encoder(self) -> LabelEncoder:
        """Load the label encoder used during training"""
        encoder_path = self.models_dir / "label_encoder.pkl"
        if encoder_path.exists():
            return joblib.load(encoder_path)
        else:
            logger.warning("Label encoder not found. Creating default encoder.")
            encoder = LabelEncoder()
            encoder.classes_ = np.array(['END', 'MIDDLE', 'NONE', 'START'])
            return encoder
    
    def load_model(self, model_name: str) -> Tuple[object, object]:
        """Load a specific model and its scaler"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Available: {self.available_models}")
            
        # Define which models need scalers
        models_with_scalers = {'logistic_regression', 'svm', 'neural_network'}
        
        # Load model
        if model_name == 'neural_network':
            model_path = self.models_dir / "neural_network_model.h5"
            model = load_model(model_path)
        elif model_name == 'xgboost':
            # Try to load the native JSON format first, fall back to pickle
            json_path = self.models_dir / "xgboost_model.json"
            if json_path.exists():
                model = xgb.XGBClassifier()
                model.load_model(str(json_path))
            else:
                pkl_path = self.models_dir / "xgboost_model.pkl"
                model = joblib.load(pkl_path)
        else:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            model = joblib.load(model_path)
            
        # Load scaler only for models that use them
        scaler = None
        if model_name in models_with_scalers:
            scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler for {model_name}")
            else:
                logger.warning(f"No scaler found for {model_name} (expected at {scaler_path})")
        else:
            logger.info(f"No scaler needed for {model_name}")
        
        logger.info(f"Loaded model: {model_name}")
        return model, scaler


class FeatureProcessor:
    """Processes test data features for prediction"""
    
    def __init__(self, training_features_path: str = "data/train/features_dataset.csv"):
        self.training_features_path = Path(training_features_path)
        self.feature_columns = self._get_training_features()
    
    def _get_training_features(self) -> List[str]:
        """Get the feature columns from training data"""
        if self.training_features_path.exists():
            train_df = pd.read_csv(self.training_features_path, nrows=1)
            # Exclude metadata and target columns - keep only features
            excluded_cols = ['Scan File_Name', 'TANAP Boundaries']
            feature_cols = [col for col in train_df.columns if col not in excluded_cols]
            logger.info(f"Found {len(feature_cols)} feature columns from training data")
            return feature_cols
        else:
            logger.warning("Training features not found. Will use all numeric columns.")
            return []
    
    def prepare_features(self, test_df: pd.DataFrame) -> np.ndarray:
        """Prepare test features for prediction"""
        # Create a copy to avoid modifying original data
        test_df_copy = test_df.copy()
        
        if self.feature_columns:
            # Test data has additional metadata columns but same feature names
            # Exclude test metadata columns and use only feature columns
            test_metadata_cols = ['Scan File_Name', 'archive_code', 'inventory_id', 'page_num', 'base_name', 'scan_url']
            available_features = [col for col in self.feature_columns if col in test_df_copy.columns]
            missing_features = [col for col in self.feature_columns if col not in test_df_copy.columns]
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features from test data: {missing_features[:5]}...")
                # Fill missing features with zeros
                for col in missing_features:
                    test_df_copy[col] = 0.0
            
            logger.info(f"Using {len(available_features)} features, {len(missing_features)} filled with zeros")
            
            # Use training feature columns in exact order
            X = test_df_copy[self.feature_columns].values
        else:
            # Fallback: use all numeric columns except metadata
            numeric_cols = test_df_copy.select_dtypes(include=[np.number]).columns
            excluded_cols = ['archive_code', 'inventory_id', 'page_num']
            feature_cols = [col for col in numeric_cols if col not in excluded_cols]
            X = test_df_copy[feature_cols].values
            logger.warning(f"No training features found, using {len(feature_cols)} available numeric columns")
            
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Prepared features shape: {X.shape}")
        return X


class BoundaryPredictor:
    """Main prediction pipeline"""
    
    def __init__(self, models_dir: str = "models"):
        self.model_loader = ModelLoader(models_dir)
        self.feature_processor = FeatureProcessor()
        
    def predict_single_model(self, test_df: pd.DataFrame, model_name: str) -> Dict:
        """Make predictions using a single model"""
        model, scaler = self.model_loader.load_model(model_name)
        X = self.feature_processor.prepare_features(test_df.copy())
        
        # Apply scaling if available
        if scaler is not None:
            X = scaler.transform(X)
            
        # Make predictions
        if model_name == 'neural_network':
            pred_probs = model.predict(X)
            predictions = np.argmax(pred_probs, axis=1)
        else:
            predictions = model.predict(X)
            if hasattr(model, 'predict_proba'):
                pred_probs = model.predict_proba(X)
            else:
                # For models without predict_proba, create dummy probabilities
                pred_probs = np.zeros((len(predictions), len(self.model_loader.label_encoder.classes_)))
                for i, pred in enumerate(predictions):
                    pred_probs[i, pred] = 1.0
        
        # Convert to boundary labels
        boundary_labels = self.model_loader.label_encoder.inverse_transform(predictions)
        
        # Get confidence scores (max probability)
        confidences = np.max(pred_probs, axis=1)
        
        return {
            'model': model_name,
            'predictions': boundary_labels,
            'probabilities': pred_probs,
            'confidences': confidences,
            'classes': self.model_loader.label_encoder.classes_
        }
    
    def predict_all_models(self, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """Make predictions using all available models"""
        results = {}
        for model_name in self.model_loader.available_models:
            try:
                results[model_name] = self.predict_single_model(test_df, model_name)
                logger.info(f"Successfully predicted with {model_name}")
            except Exception as e:
                logger.error(f"Failed to predict with {model_name}: {e}")
                
        return results
    
    def format_output(self, test_df: pd.DataFrame, predictions: Dict, output_format: str = 'csv') -> pd.DataFrame:
        """Format prediction results for output"""
        # Include test metadata columns if available
        metadata_cols = ['Scan File_Name']
        optional_metadata = ['archive_code', 'inventory_id', 'page_num', 'base_name', 'scan_url']
        
        for col in optional_metadata:
            if col in test_df.columns:
                metadata_cols.append(col)
        
        output_df = test_df[metadata_cols].copy()
        
        if len(predictions) == 1:
            # Single model results
            model_name, result = list(predictions.items())[0]
            output_df[f'predicted_boundary'] = result['predictions']
            output_df[f'confidence'] = result['confidences']
            
            # Add probability columns
            for i, class_name in enumerate(result['classes']):
                output_df[f'prob_{class_name}'] = result['probabilities'][:, i]
                
        else:
            # Multiple model results
            for model_name, result in predictions.items():
                output_df[f'{model_name}_prediction'] = result['predictions']
                output_df[f'{model_name}_confidence'] = result['confidences']
                
                # Add top probability for each model
                output_df[f'{model_name}_max_prob'] = np.max(result['probabilities'], axis=1)
        
        # Add metadata
        output_df['prediction_timestamp'] = datetime.now().isoformat()
        output_df['total_models_used'] = len(predictions)
        
        return output_df


def main():
    parser = argparse.ArgumentParser(description='Predict TANAP boundaries on test inventories')
    parser.add_argument('--model', type=str, choices=['logistic_regression', 'random_forest', 'xgboost', 'svm', 'neural_network', 'all'], 
                       default='all', help='Model to use for prediction')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file or directory')
    parser.add_argument('--output', type=str, required=True, help='Output file or directory')
    parser.add_argument('--format', type=str, choices=['csv', 'json'], default='csv', help='Output format')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory containing trained models')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize predictor
    predictor = BoundaryPredictor(args.models_dir)
    
    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Single file
        test_files = [input_path]
    elif input_path.is_dir():
        # Directory - find all CSV files
        test_files = list(input_path.glob("*.csv"))
        if not test_files:
            logger.error(f"No CSV files found in {input_path}")
            sys.exit(1)
    else:
        logger.error(f"Input path {input_path} does not exist")
        sys.exit(1)
    
    logger.info(f"Found {len(test_files)} test files to process")
    
    # Process each file
    for test_file in test_files:
        logger.info(f"Processing {test_file}")
        
        try:
            # Load test data
            test_df = pd.read_csv(test_file)
            logger.info(f"Loaded {len(test_df)} samples from {test_file}")
            
            # Make predictions
            if args.model == 'all':
                predictions = predictor.predict_all_models(test_df)
            else:
                predictions = {args.model: predictor.predict_single_model(test_df, args.model)}
            
            # Format output
            output_df = predictor.format_output(test_df, predictions, args.format)
            
            # Save results
            if output_path.is_dir():
                output_file = output_path / f"{test_file.stem}_predictions.{args.format}"
            else:
                output_file = output_path
                
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if args.format == 'csv':
                output_df.to_csv(output_file, index=False)
            else:  # json
                output_df.to_json(output_file, orient='records', indent=2)
                
            logger.info(f"Saved predictions to {output_file}")
            
            # Print summary
            for model_name in predictions:
                pred_counts = pd.Series(predictions[model_name]['predictions']).value_counts()
                logger.info(f"{model_name} predictions: {dict(pred_counts)}")
                
        except Exception as e:
            logger.error(f"Failed to process {test_file}: {e}")
            continue
    
    logger.info("Prediction complete!")


if __name__ == "__main__":
    main()