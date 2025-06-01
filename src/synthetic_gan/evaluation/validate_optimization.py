#!/usr/bin/env python3
"""
Validate that optimization preserves data quality
Compare original vs optimized synthetic data
"""

import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def compare_data_quality(original_file, optimized_file):
    """
    Compare original and optimized data to ensure quality preservation
    """
    print("🔍 VALIDATING DATA OPTIMIZATION QUALITY")
    print("="*60)
    
    # Load original data
    print("Loading original data...")
    df_orig = pd.read_csv(original_file)
    
    # Load optimized data
    print("Loading optimized data...")
    if optimized_file.endswith('.gz'):
        with gzip.open(optimized_file, 'rt') as f:
            df_opt = pd.read_csv(f)
    else:
        df_opt = pd.read_csv(optimized_file)
    
    print(f"✓ Original shape: {df_orig.shape}")
    print(f"✓ Optimized shape: {df_opt.shape}")
    
    # Basic structure validation
    assert df_orig.shape == df_opt.shape, "Shape mismatch!"
    assert list(df_orig.columns) == list(df_opt.columns), "Column mismatch!"
    print("✅ Structure validation passed")
    
    # Statistical comparison
    print("\n📊 STATISTICAL COMPARISON")
    print("-" * 40)
    
    results = {
        'column': [],
        'mean_diff': [],
        'std_diff': [],
        'correlation': [],
        'max_abs_diff': []
    }
    
    significant_changes = []
    
    for col in df_orig.columns:
        if df_orig[col].dtype in ['int64', 'float64'] and df_opt[col].dtype in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            # Numeric column comparison
            orig_vals = df_orig[col].dropna()
            opt_vals = df_opt[col].dropna()
            
            if len(orig_vals) > 0 and len(opt_vals) > 0:
                mean_diff = abs(orig_vals.mean() - opt_vals.mean())
                std_diff = abs(orig_vals.std() - opt_vals.std())
                correlation = np.corrcoef(orig_vals, opt_vals)[0, 1] if len(orig_vals) == len(opt_vals) else np.nan
                max_abs_diff = abs(orig_vals - opt_vals).max() if len(orig_vals) == len(opt_vals) else np.nan
                
                results['column'].append(col)
                results['mean_diff'].append(mean_diff)
                results['std_diff'].append(std_diff)
                results['correlation'].append(correlation)
                results['max_abs_diff'].append(max_abs_diff)
                
                # Check for significant changes
                if mean_diff > orig_vals.std() * 0.01:  # More than 1% of std dev
                    significant_changes.append(f"{col}: Mean diff {mean_diff:.6f}")
        else:
            # Categorical column comparison
            if not df_orig[col].equals(df_opt[col]):
                significant_changes.append(f"{col}: Categorical values changed")
    
    # Summary statistics
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        print(f"📈 Numerical columns analyzed: {len(df_results)}")
        print(f"   Mean correlation: {df_results['correlation'].mean():.6f}")
        print(f"   Max mean difference: {df_results['mean_diff'].max():.6f}")
        print(f"   Max std difference: {df_results['std_diff'].max():.6f}")
        print(f"   Max absolute difference: {df_results['max_abs_diff'].max():.6f}")
    
    # Distribution comparison for key columns
    print("\n📊 DISTRIBUTION COMPARISON (First 5 numeric columns)")
    print("-" * 50)
    
    numeric_cols = [col for col in df_orig.columns if df_orig[col].dtype in ['int64', 'float64']][:5]
    
    for col in numeric_cols:
        # KS test for distribution similarity
        if df_orig[col].notna().sum() > 100 and df_opt[col].notna().sum() > 100:
            ks_stat, p_value = stats.ks_2samp(
                df_orig[col].dropna().sample(min(1000, len(df_orig[col].dropna()))),
                df_opt[col].dropna().sample(min(1000, len(df_opt[col].dropna())))
            )
            print(f"  {col}: KS p-value = {p_value:.6f} {'✅' if p_value > 0.01 else '⚠'}")
    
    # Final assessment
    print("\n🎯 QUALITY ASSESSMENT")
    print("-" * 30)
    
    if len(significant_changes) == 0:
        print("✅ EXCELLENT: No significant changes detected")
        quality_score = "EXCELLENT"
    elif len(significant_changes) <= 3:
        print(f"✅ GOOD: Minor changes in {len(significant_changes)} columns")
        for change in significant_changes:
            print(f"   - {change}")
        quality_score = "GOOD"
    else:
        print(f"⚠ CAUTION: Changes detected in {len(significant_changes)} columns")
        for change in significant_changes[:5]:  # Show first 5
            print(f"   - {change}")
        if len(significant_changes) > 5:
            print(f"   ... and {len(significant_changes) - 5} more")
        quality_score = "CAUTION"
    
    # File size comparison
    orig_size = pd.read_csv(original_file).memory_usage(deep=True).sum() / (1024**2)
    if optimized_file.endswith('.gz'):
        import os
        opt_size = os.path.getsize(optimized_file) / (1024**2)
        compression_info = f"File size: {orig_size:.1f}MB → {opt_size:.1f}MB (compressed)"
    else:
        opt_size = pd.read_csv(optimized_file).memory_usage(deep=True).sum() / (1024**2)
        compression_info = f"Memory usage: {orig_size:.1f}MB → {opt_size:.1f}MB"
    
    print(f"\n💾 {compression_info}")
    
    return quality_score, df_results

def main():
    original_file = "synthetic_data.csv"
    optimized_file = "synthetic_data_submission.csv.gz"
    
    try:
        quality_score, results = compare_data_quality(original_file, optimized_file)
        
        print("\n" + "="*60)
        if quality_score in ["EXCELLENT", "GOOD"]:
            print("🎉 VALIDATION PASSED!")
            print("✅ Optimized data maintains high quality")
            print("✅ Safe to submit for competition")
        else:
            print("⚠ VALIDATION CONCERNS")
            print("Review the changes above before submission")
        print("="*60)
        
        return 0 if quality_score in ["EXCELLENT", "GOOD"] else 1
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 