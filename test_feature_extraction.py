#!/usr/bin/env python3
"""
Test Feature Extraction Pipeline for TANAP Boundaries Prediction

This script processes test data files from /data/test/unseen testsets/ directory
and generates the same feature format as the training pipeline, but without target column.

Key differences from training pipeline:
- Handles test data structure (archive_code, inventory_id, page_num, base_name columns)
- No TANAP Boundaries target column
- Groups by inventory_id for sequence feature processing
- Preserves metadata for prediction pipeline
- Can process single file or batch process directory
- CLI interface with progress tracking

Author: Generated for Document Segmentation Project
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

# Import feature extractors from the existing feature_extraction.py
from feature_extraction import XMLLayoutFeatureExtractor, XMINERFeatureExtractor, SequenceFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestSequenceFeatureExtractor(SequenceFeatureExtractor):
    """Modified sequence feature extractor for test data that groups by inventory_id"""
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features that consider the sequence context, grouped by inventory_id"""
        df = df.copy()
        
        # Sort by inventory_id and page_num to ensure proper sequence order within each inventory
        df = df.sort_values(['inventory_id', 'page_num']).reset_index(drop=True)
        
        # Initialize new feature columns
        sequence_columns = []
        
        # Previous and next page features (within same inventory)
        for col in ['num_text_regions', 'num_text_lines', 'total_text_area', 'num_sentences', 'num_tokens']:
            if col in df.columns:
                # Initialize columns
                df[f'prev_{col}'] = 0.0
                df[f'next_{col}'] = 0.0
                df[f'delta_prev_{col}'] = 0.0
                df[f'delta_next_{col}'] = 0.0
                
                # Calculate features within each inventory group
                for inventory_id in df['inventory_id'].unique():
                    mask = df['inventory_id'] == inventory_id
                    inventory_df = df[mask].copy()
                    
                    if len(inventory_df) > 1:
                        # Previous page features
                        prev_values = inventory_df[col].shift(1).fillna(0)
                        df.loc[mask, f'prev_{col}'] = prev_values
                        
                        # Next page features
                        next_values = inventory_df[col].shift(-1).fillna(0)
                        df.loc[mask, f'next_{col}'] = next_values
                        
                        # Delta features
                        df.loc[mask, f'delta_prev_{col}'] = inventory_df[col] - prev_values
                        df.loc[mask, f'delta_next_{col}'] = next_values - inventory_df[col]
                
                sequence_columns.extend([f'prev_{col}', f'next_{col}', f'delta_prev_{col}', f'delta_next_{col}'])
        
        # Rolling window features (3-page window within same inventory)
        window_size = 3
        for col in ['num_text_regions', 'total_text_area', 'num_sentences']:
            if col in df.columns:
                df[f'rolling_mean_{col}'] = 0.0
                df[f'rolling_std_{col}'] = 0.0
                
                for inventory_id in df['inventory_id'].unique():
                    mask = df['inventory_id'] == inventory_id
                    inventory_df = df[mask].copy()
                    
                    if len(inventory_df) >= window_size:
                        rolling_mean = inventory_df[col].rolling(window=window_size, center=True).mean()
                        rolling_std = inventory_df[col].rolling(window=window_size, center=True).std()
                        
                        df.loc[mask, f'rolling_mean_{col}'] = rolling_mean.fillna(inventory_df[col])
                        df.loc[mask, f'rolling_std_{col}'] = rolling_std.fillna(0)
                    else:
                        # For small inventories, use the column values as rolling mean
                        df.loc[mask, f'rolling_mean_{col}'] = inventory_df[col]
                        df.loc[mask, f'rolling_std_{col}'] = 0
                
                sequence_columns.extend([f'rolling_mean_{col}', f'rolling_std_{col}'])
        
        # Position in document sequence features (within inventory)
        df['position_in_sequence'] = 0
        df['position_normalized'] = 0.0
        
        for inventory_id in df['inventory_id'].unique():
            mask = df['inventory_id'] == inventory_id
            inventory_size = mask.sum()
            
            df.loc[mask, 'position_in_sequence'] = range(inventory_size)
            df.loc[mask, 'position_normalized'] = np.arange(inventory_size) / max(1, inventory_size - 1) if inventory_size > 1 else 0.0
        
        sequence_columns.extend(['position_in_sequence', 'position_normalized'])
        
        logger.info(f"Generated {len(sequence_columns)} sequence features")
        return df


def load_test_data(file_path: str) -> pd.DataFrame:
    """Load a single test data file"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples from {file_path}")
        
        # Verify required columns exist
        required_columns = ['Scan File_Name', 'archive_code', 'inventory_id', 'page_num', 'base_name', 'xml_data', 'xmi_data', 'scan_url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Convert page_num to int for proper sorting
        df['page_num'] = pd.to_numeric(df['page_num'], errors='coerce').fillna(0).astype(int)
        
        logger.info(f"Inventory IDs in file: {sorted(df['inventory_id'].unique())}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise


def load_multiple_test_files(directory_path: str) -> pd.DataFrame:
    """Load and combine multiple test data files from directory"""
    test_files = glob.glob(os.path.join(directory_path, "unseen_test_set_*.csv"))
    
    if not test_files:
        raise ValueError(f"No test files found in {directory_path}")
    
    logger.info(f"Found {len(test_files)} test files")
    
    all_dfs = []
    for file_path in sorted(test_files):
        df = load_test_data(file_path)
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} samples from {len(all_dfs)} files")
    logger.info(f"Total inventory IDs: {sorted(combined_df['inventory_id'].unique())}")
    
    return combined_df


def extract_test_features(df: pd.DataFrame, batch_size: int = 500) -> pd.DataFrame:
    """Extract features from test data using existing extractors"""
    
    # Initialize feature extractors
    xml_extractor = XMLLayoutFeatureExtractor()
    xmi_extractor = XMINERFeatureExtractor()
    sequence_extractor = TestSequenceFeatureExtractor()
    
    # Extract XML layout features
    logger.info("Extracting XML layout features...")
    xml_features = []
    
    for idx in range(0, len(df), batch_size):
        end_idx = min(idx + batch_size, len(df))
        batch = df.iloc[idx:end_idx]
        
        logger.info(f"Processing XML batch {idx//batch_size + 1}/{(len(df)-1)//batch_size + 1} ({idx+1}-{end_idx}/{len(df)})")
        
        batch_features = []
        for _, row in batch.iterrows():
            features = xml_extractor.extract_features(row['xml_data'])
            batch_features.append(features)
        
        xml_features.extend(batch_features)
    
    # Extract XMI NER features
    logger.info("Extracting XMI NER features...")
    xmi_features = []
    
    for idx in range(0, len(df), batch_size):
        end_idx = min(idx + batch_size, len(df))
        batch = df.iloc[idx:end_idx]
        
        logger.info(f"Processing XMI batch {idx//batch_size + 1}/{(len(df)-1)//batch_size + 1} ({idx+1}-{end_idx}/{len(df)})")
        
        batch_features = []
        for _, row in batch.iterrows():
            features = xmi_extractor.extract_features(row['xmi_data'])
            batch_features.append(features)
        
        xmi_features.extend(batch_features)
    
    # Combine features
    logger.info("Combining features...")
    xml_df = pd.DataFrame(xml_features)
    xmi_df = pd.DataFrame(xmi_features)
    
    # Create feature dataset preserving metadata
    feature_df = pd.concat([
        df[['Scan File_Name', 'archive_code', 'inventory_id', 'page_num', 'base_name', 'scan_url']],
        xml_df,
        xmi_df
    ], axis=1)
    
    # Extract sequence features
    logger.info("Extracting sequence features...")
    feature_df = sequence_extractor.extract_features(feature_df)
    
    logger.info(f"Feature extraction completed! Final shape: {feature_df.shape}")
    
    return feature_df


def save_test_features(df: pd.DataFrame, output_path: str, include_metadata: bool = True):
    """Save extracted features to CSV"""
    
    if include_metadata:
        # Save with metadata for prediction pipeline
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features with metadata to: {output_path}")
    else:
        # Save without metadata (features only) for direct model input
        metadata_cols = ['Scan File_Name', 'archive_code', 'inventory_id', 'page_num', 'base_name', 'scan_url']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        features_only_path = output_path.replace('.csv', '_features_only.csv')
        df[feature_cols].to_csv(features_only_path, index=False)
        logger.info(f"Saved features only to: {features_only_path}")
    
    # Print summary statistics
    feature_cols = [col for col in df.columns if col not in ['Scan File_Name', 'archive_code', 'inventory_id', 'page_num', 'base_name', 'scan_url']]
    logger.info(f"Generated {len(feature_cols)} features")
    
    # Show first 10 feature statistics
    logger.info("\nFeature Summary (first 10 features):")
    for col in feature_cols[:10]:
        if df[col].dtype in ['int64', 'float64']:
            logger.info(f"{col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, min={df[col].min():.3f}, max={df[col].max():.3f}")
        else:
            logger.info(f"{col}: {df[col].dtype}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Extract features from test data files')
    parser.add_argument('--input', '-i', required=True,
                        help='Input file path or directory path containing test files')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file path for features')
    parser.add_argument('--batch-size', '-b', type=int, default=500,
                        help='Batch size for processing (default: 500)')
    parser.add_argument('--no-metadata', action='store_true',
                        help='Save features without metadata columns')
    
    args = parser.parse_args()
    
    logger.info("Starting test feature extraction pipeline...")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    try:
        # Load test data
        if os.path.isfile(args.input):
            logger.info(f"Loading single test file: {args.input}")
            df = load_test_data(args.input)
        elif os.path.isdir(args.input):
            logger.info(f"Loading test files from directory: {args.input}")
            df = load_multiple_test_files(args.input)
        else:
            raise ValueError(f"Input path does not exist: {args.input}")
        
        # Extract features
        feature_df = extract_test_features(df, batch_size=args.batch_size)
        
        # Save results
        save_test_features(feature_df, args.output, include_metadata=not args.no_metadata)
        
        logger.info("Test feature extraction completed successfully!")
        
        # Final statistics
        logger.info(f"\nFinal Statistics:")
        logger.info(f"Total samples processed: {len(feature_df)}")
        logger.info(f"Total inventories: {feature_df['inventory_id'].nunique()}")
        logger.info(f"Inventory distribution:")
        for inv_id, count in feature_df['inventory_id'].value_counts().sort_index().items():
            logger.info(f"  {inv_id}: {count} pages")
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        raise


if __name__ == "__main__":
    main()