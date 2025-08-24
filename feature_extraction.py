#!/usr/bin/env python3
"""
Feature Extraction Pipeline for TANAP Boundaries Prediction

This script extracts meaningful features from XML (layout) and XMI (NER) data
to create a training dataset for predicting document boundary sequences.

Author: Generated for Document Segmentation Project
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XMLLayoutFeatureExtractor:
    """Extract features from PAGE XML layout data"""
    
    def __init__(self):
        self.namespace = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    
    def extract_features(self, xml_string: str) -> Dict:
        """Extract layout-based features from XML data"""
        features = {
            # Basic layout metrics
            'num_text_regions': 0,
            'num_text_lines': 0,
            'num_words': 0,
            'avg_region_area': 0,
            'total_text_area': 0,
            'page_coverage_ratio': 0,
            
            # Spatial distribution features
            'regions_top_half': 0,
            'regions_bottom_half': 0,
            'regions_left_half': 0,
            'regions_right_half': 0,
            
            # Text region types
            'header_regions': 0,
            'paragraph_regions': 0,
            'list_regions': 0,
            'table_regions': 0,
            'other_regions': 0,
            
            # Layout complexity
            'region_area_variance': 0,
            'vertical_spacing_avg': 0,
            'horizontal_spacing_avg': 0,
            
            # Reading order features
            'has_reading_order': 0,
            'reading_order_complexity': 0,
        }
        
        try:
            root = ET.fromstring(xml_string)
            
            # Get page dimensions
            page = root.find('.//page:Page', self.namespace)
            page_width = page_height = 1
            if page is not None:
                page_width = int(page.get('imageWidth', 1))
                page_height = int(page.get('imageHeight', 1))
            
            # Extract text regions
            text_regions = root.findall('.//page:TextRegion', self.namespace)
            features['num_text_regions'] = len(text_regions)
            
            if text_regions:
                region_areas = []
                region_centers = []
                
                for region in text_regions:
                    # Count text lines and words in this region
                    text_lines = region.findall('.//page:TextLine', self.namespace)
                    words = region.findall('.//page:Word', self.namespace)
                    features['num_text_lines'] += len(text_lines)
                    features['num_words'] += len(words)
                    
                    # Get region coordinates and calculate area
                    coords = region.find('.//page:Coords', self.namespace)
                    if coords is not None:
                        area, center = self._calculate_region_metrics(coords.get('points', ''), page_width, page_height)
                        region_areas.append(area)
                        region_centers.append(center)
                    
                    # Count region types
                    custom_attr = region.get('custom', '')
                    if 'header' in custom_attr.lower():
                        features['header_regions'] += 1
                    elif 'paragraph' in custom_attr.lower():
                        features['paragraph_regions'] += 1
                    elif 'list' in custom_attr.lower():
                        features['list_regions'] += 1
                    elif 'table' in custom_attr.lower():
                        features['table_regions'] += 1
                    else:
                        features['other_regions'] += 1
                
                # Calculate spatial features
                if region_areas:
                    features['avg_region_area'] = np.mean(region_areas)
                    features['total_text_area'] = sum(region_areas)
                    features['page_coverage_ratio'] = features['total_text_area'] / (page_width * page_height)
                    features['region_area_variance'] = np.var(region_areas)
                
                # Calculate spatial distribution
                if region_centers:
                    for center_x, center_y in region_centers:
                        if center_y < page_height / 2:
                            features['regions_top_half'] += 1
                        else:
                            features['regions_bottom_half'] += 1
                        
                        if center_x < page_width / 2:
                            features['regions_left_half'] += 1
                        else:
                            features['regions_right_half'] += 1
            
            # Check for reading order
            reading_order = root.find('.//page:ReadingOrder', self.namespace)
            if reading_order is not None:
                features['has_reading_order'] = 1
                ordered_groups = reading_order.findall('.//page:OrderedGroup', self.namespace)
                features['reading_order_complexity'] = len(ordered_groups)
                
        except Exception as e:
            logger.warning(f"Error processing XML: {str(e)}")
            
        return features
    
    def _calculate_region_metrics(self, points_str: str, page_width: int, page_height: int) -> Tuple[float, Tuple[float, float]]:
        """Calculate area and center point from coordinate string"""
        try:
            points = []
            coords = points_str.split()
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x, y = int(coords[i]), int(coords[i + 1])
                    points.append((x, y))
            
            if len(points) < 3:
                return 0, (0, 0)
            
            # Calculate area using shoelace formula
            area = 0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            area = abs(area) / 2
            
            # Calculate center
            center_x = sum(p[0] for p in points) / len(points)
            center_y = sum(p[1] for p in points) / len(points)
            
            return area, (center_x, center_y)
            
        except Exception:
            return 0, (0, 0)


class XMINERFeatureExtractor:
    """Extract features from XMI NER data"""
    
    def extract_features(self, xmi_string: str) -> Dict:
        """Extract NER and linguistic features from XMI data"""
        features = {
            # Basic NER counts
            'num_sentences': 0,
            'num_tokens': 0,
            'num_named_entities': 0,
            
            # Entity types (common NER categories)
            'person_entities': 0,
            'location_entities': 0,
            'organization_entities': 0,
            'date_entities': 0,
            'money_entities': 0,
            'misc_entities': 0,
            
            # Text characteristics
            'avg_sentence_length': 0,
            'avg_token_length': 0,
            'token_length_variance': 0,
            
            # Document structure indicators
            'has_headings': 0,
            'num_headings': 0,
            'has_punctuation': 0,
            'punctuation_density': 0,
            
            # Language features
            'dutch_indicators': 0,
            'formal_language_score': 0,
            'numeric_content_ratio': 0,
        }
        
        if pd.isna(xmi_string) or not xmi_string.strip():
            return features
        
        try:
            root = ET.fromstring(xmi_string)
            
            # Count basic elements
            sentences = root.findall('.//type:Sentence', {'type': 'http:///de/tudarmstadt/ukp/dkpro/core/api/segmentation/type.ecore'})
            tokens = root.findall('.//type:Token', {'type': 'http:///de/tudarmstadt/ukp/dkpro/core/api/segmentation/type.ecore'})
            headings = root.findall('.//type:Heading', {'type': 'http:///de/tudarmstadt/ukp/dkpro/core/api/segmentation/type.ecore'})
            
            features['num_sentences'] = len(sentences)
            features['num_tokens'] = len(tokens)
            features['num_headings'] = len(headings)
            features['has_headings'] = 1 if headings else 0
            
            # Extract text content from sofa
            sofa = root.find('.//cas:Sofa', {'cas': 'http:///uima/cas.ecore'})
            text_content = ""
            if sofa is not None:
                text_content = sofa.get('sofaString', '')
                # Decode HTML entities
                text_content = text_content.replace('&#10;', '\n').replace('&quot;', '"').replace('&amp;', '&')
            
            # Analyze text content
            if text_content:
                # Calculate text metrics
                sentences_text = [s.strip() for s in text_content.split('.') if s.strip()]
                if sentences_text:
                    features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences_text])
                
                words = text_content.split()
                if words:
                    word_lengths = [len(w) for w in words]
                    features['avg_token_length'] = np.mean(word_lengths)
                    features['token_length_variance'] = np.var(word_lengths)
                
                # Count numeric content
                numeric_chars = sum(1 for c in text_content if c.isdigit())
                features['numeric_content_ratio'] = numeric_chars / len(text_content) if text_content else 0
                
                # Count punctuation
                punctuation_chars = sum(1 for c in text_content if c in '.,;:!?()-[]{}')
                features['punctuation_density'] = punctuation_chars / len(text_content) if text_content else 0
                features['has_punctuation'] = 1 if punctuation_chars > 0 else 0
                
                # Dutch language indicators (simple heuristics)
                dutch_words = ['de', 'het', 'een', 'van', 'en', 'in', 'op', 'voor', 'met', 'aan']
                dutch_count = sum(1 for word in words if word.lower() in dutch_words)
                features['dutch_indicators'] = dutch_count / len(words) if words else 0
                
                # Formal language score (based on sentence length and vocabulary)
                avg_word_length = features['avg_token_length']
                avg_sent_length = features['avg_sentence_length']
                features['formal_language_score'] = min(1.0, (avg_word_length * avg_sent_length) / 100)
            
            # Look for named entity annotations (namespace-agnostic)
            for elem in root.iter():
                tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                if 'Person' in tag_name or 'PERSON' in tag_name:
                    features['person_entities'] += 1
                elif 'Location' in tag_name or 'LOCATION' in tag_name or 'Place' in tag_name:
                    features['location_entities'] += 1
                elif 'Organization' in tag_name or 'ORGANIZATION' in tag_name:
                    features['organization_entities'] += 1
                elif 'Date' in tag_name or 'TIME' in tag_name:
                    features['date_entities'] += 1
                elif 'Money' in tag_name or 'MONEY' in tag_name:
                    features['money_entities'] += 1
                elif 'Entity' in tag_name or 'NE' in tag_name:
                    features['misc_entities'] += 1
            
            features['num_named_entities'] = (features['person_entities'] + features['location_entities'] + 
                                           features['organization_entities'] + features['date_entities'] + 
                                           features['money_entities'] + features['misc_entities'])
            
        except Exception as e:
            logger.warning(f"Error processing XMI: {str(e)}")
            
        return features


class SequenceFeatureExtractor:
    """Extract sequence-based features for document boundary prediction"""
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features that consider the sequence context"""
        df = df.copy()
        
        # Sort by scan file name to ensure proper sequence order
        df = df.sort_values('Scan File_Name').reset_index(drop=True)
        
        # Previous and next page features
        for col in ['num_text_regions', 'num_text_lines', 'total_text_area', 'num_sentences', 'num_tokens']:
            if col in df.columns:
                df[f'prev_{col}'] = df[col].shift(1).fillna(0)
                df[f'next_{col}'] = df[col].shift(-1).fillna(0)
                df[f'delta_prev_{col}'] = df[col] - df[f'prev_{col}']
                df[f'delta_next_{col}'] = df[f'next_{col}'] - df[col]
        
        # Rolling window features (3-page window)
        window_size = 3
        for col in ['num_text_regions', 'total_text_area', 'num_sentences']:
            if col in df.columns:
                df[f'rolling_mean_{col}'] = df[col].rolling(window=window_size, center=True).mean().fillna(df[col])
                df[f'rolling_std_{col}'] = df[col].rolling(window=window_size, center=True).std().fillna(0)
        
        # Position in document sequence features
        df['position_in_sequence'] = range(len(df))
        df['position_normalized'] = df['position_in_sequence'] / len(df)
        
        return df


def main():
    """Main function to run the feature extraction pipeline"""
    logger.info("Starting feature extraction pipeline...")
    
    # Load the dataset
    logger.info("Loading dataset...")
    input_file = "data/train/renate_dataset.csv"
    df = pd.read_csv(input_file)
    
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"TANAP Boundaries distribution:\n{df['TANAP Boundaries'].value_counts()}")
    
    # Initialize feature extractors
    xml_extractor = XMLLayoutFeatureExtractor()
    xmi_extractor = XMINERFeatureExtractor()
    sequence_extractor = SequenceFeatureExtractor()
    
    # Extract features
    logger.info("Extracting XML layout features...")
    xml_features = []
    for idx, xml_data in enumerate(df['xml_data']):
        if idx % 1000 == 0:
            logger.info(f"Processed {idx}/{len(df)} XML samples")
        features = xml_extractor.extract_features(xml_data)
        xml_features.append(features)
    
    logger.info("Extracting XMI NER features...")
    xmi_features = []
    for idx, xmi_data in enumerate(df['xmi_data']):
        if idx % 1000 == 0:
            logger.info(f"Processed {idx}/{len(df)} XMI samples")
        features = xmi_extractor.extract_features(xmi_data)
        xmi_features.append(features)
    
    # Combine features
    logger.info("Combining features...")
    xml_df = pd.DataFrame(xml_features)
    xmi_df = pd.DataFrame(xmi_features)
    
    # Create final feature dataset
    feature_df = pd.concat([
        df[['Scan File_Name', 'TANAP Boundaries']],
        xml_df,
        xmi_df
    ], axis=1)
    
    # Extract sequence features
    logger.info("Extracting sequence features...")
    feature_df = sequence_extractor.extract_features(feature_df)
    
    # Save the feature dataset
    output_file = "data/train/features_dataset.csv"
    feature_df.to_csv(output_file, index=False)
    
    logger.info(f"Feature extraction completed!")
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"Final dataset shape: {feature_df.shape}")
    logger.info(f"Feature columns: {list(feature_df.columns)}")
    
    # Print feature summary
    logger.info("\nFeature Summary:")
    feature_cols = [col for col in feature_df.columns if col not in ['Scan File_Name', 'TANAP Boundaries']]
    for col in feature_cols[:10]:  # Show first 10 features
        logger.info(f"{col}: mean={feature_df[col].mean():.3f}, std={feature_df[col].std():.3f}")
    
    # Show class distribution
    logger.info(f"\nFinal TANAP Boundaries distribution:\n{feature_df['TANAP Boundaries'].value_counts()}")


if __name__ == "__main__":
    main()