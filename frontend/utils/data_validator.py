"""
Data Validation Utilities
Validates CSV structure and provides helpful feedback
"""

import pandas as pd
import streamlit as st

class DataValidator:
    """Validates uploaded CSV data"""
    
    # Required columns for basic analysis
    REQUIRED_COLUMNS = [
        'Track Name',
        'Artist Name(s)',
        'Duration (ms)'
    ]
    
    # Optional but recommended columns
    RECOMMENDED_COLUMNS = [
        'Popularity',
        'Release Date',
        'Added At',
        'Danceability',
        'Energy',
        'Valence',
        'Acousticness',
        'Tempo'
    ]
    
    @staticmethod
    def validate_csv(df):
        """
        Validate CSV structure and content
        
        Returns:
            (is_valid, issues, warnings)
        """
        issues = []
        warnings = []
        
        # Check if dataframe is empty
        if len(df) == 0:
            issues.append("CSV file is empty")
            return False, issues, warnings
        
        # Check required columns
        missing_required = [col for col in DataValidator.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_required:
            issues.append(f"Missing required columns: {', '.join(missing_required)}")
            return False, issues, warnings
        
        # Check recommended columns
        missing_recommended = [col for col in DataValidator.RECOMMENDED_COLUMNS if col not in df.columns]
        
        if missing_recommended:
            warnings.append(f"Missing recommended columns: {', '.join(missing_recommended)}")
            warnings.append("Some features may not be available without these columns")
        
        # Check for empty values in key columns
        for col in DataValidator.REQUIRED_COLUMNS:
            if df[col].isna().sum() > 0:
                null_count = df[col].isna().sum()
                warnings.append(f"Column '{col}' has {null_count} empty values")
        
        # Check data types
        if 'Duration (ms)' in df.columns:
            try:
                pd.to_numeric(df['Duration (ms)'], errors='coerce')
            except:
                warnings.append("Duration (ms) column contains non-numeric values")
        
        if 'Popularity' in df.columns:
            try:
                pd.to_numeric(df['Popularity'], errors='coerce')
            except:
                warnings.append("Popularity column contains non-numeric values")
        
        # Check date columns
        date_cols = ['Added At', 'Release Date']
        for col in date_cols:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='coerce')
                except:
                    warnings.append(f"{col} column has invalid date format")
        
        return True, issues, warnings
    
    @staticmethod
    def get_column_info(df):
        """Get information about available columns"""
        info = {
            'total_columns': len(df.columns),
            'available_features': [],
            'missing_features': []
        }
        
        # Check which features are available
        feature_map = {
            'Top Tracks': ['Track Name', 'Popularity'],
            'Top Artists': ['Artist Name(s)'],
            'Playlist Age': ['Added At'],
            'Listening Age': ['Release Date'],
            'Temporal Analysis': ['Added At'],
            'Mood Analysis': ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Tempo'],
            'Popularity Distribution': ['Popularity'],
            'Audio Features': ['Danceability', 'Energy', 'Valence', 'Acousticness']
        }
        
        for feature, required_cols in feature_map.items():
            if all(col in df.columns for col in required_cols):
                info['available_features'].append(feature)
            else:
                info['missing_features'].append(feature)
        
        return info
    
    @staticmethod
    def display_validation_results(is_valid, issues, warnings):
        """Display validation results in Streamlit"""
        if not is_valid:
            st.error("❌ CSV Validation Failed")
            for issue in issues:
                st.error(f"• {issue}")
            return False
        
        if warnings:
            with st.expander("⚠️ Validation Warnings", expanded=False):
                for warning in warnings:
                    st.warning(f"• {warning}")
        
        return True
    
    @staticmethod
    def suggest_fixes(issues, warnings):
        """Provide suggestions to fix validation issues"""
        suggestions = []
        
        for issue in issues:
            if "Missing required columns" in issue:
                suggestions.append("Make sure you're exporting from Exportify with all track details enabled")
            if "empty" in issue.lower():
                suggestions.append("Your CSV file appears to be empty. Try exporting your playlist again")
        
        for warning in warnings:
            if "Missing recommended columns" in warning:
                suggestions.append("Enable 'Audio Features' when exporting from Exportify")
            if "empty values" in warning:
                suggestions.append("Some tracks may have incomplete data - this is normal for older playlists")
        
        return suggestions