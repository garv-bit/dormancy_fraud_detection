"""
Main execution script for Financial Fraud Detection EDA
Run this script to perform complete analysis
"""

import sys
from pathlib import Path

# Import all components
from Config import EDAConfig
from Fraud_eda import FraudEDA


def main():
    """Main execution function"""
    
    # Create configuration (can customize here or use defaults)
    config = EDAConfig()
    
    # Optionally customize config
    # config.INPUT_CSV = "your_custom_file.csv"
    # config.CORRELATION_THRESHOLD = 0.7
    # config.N_ESTIMATORS = 200
    
    print("=" * 120)
    print(" " * 40 + "FINANCIAL FRAUD DETECTION EDA")
    print(" " * 48 + "Production Version 2.0")
    print("=" * 120)
    print(f"\n📋 Configuration:")
    print(f"   Input File: {config.INPUT_CSV}")
    print(f"   Output File: {config.OUTPUT_PICKLE}")
    print(f"   Log File: {config.LOG_FILE}")
    print(f"   Correlation Threshold: {config.CORRELATION_THRESHOLD}")
    print(f"   PCA Variance Threshold: {config.PCA_VARIANCE_THRESHOLD}")
    print(f"   Random State: {config.RANDOM_STATE}")
    print("\n" + "=" * 120)
    
    try:
        # Initialize EDA pipeline
        eda = FraudEDA(config)
        
        # Run complete analysis
        results = eda.run_full_analysis()
        
        # Save results
        eda.save_results()
        
        # Print summary
        print("\n" + "=" * 120)
        print("ANALYSIS SUMMARY")
        print("=" * 120)
        
        metadata = results.get('metadata', {})
        quality = results.get('data_quality', {})
        model = results.get('model_readiness', {})
        temporal = results.get('temporal_analysis', {})
        
        print(f"\n📊 Dataset Information:")
        print(f"   • Total Transactions: {metadata.get('total_rows', 'N/A'):,}")
        print(f"   • Total Features: {metadata.get('total_columns', 'N/A')}")
        print(f"   • Memory Usage: {metadata.get('memory_mb', 0):.2f} MB")
        
        print(f"\n✅ Data Quality:")
        overall_quality = quality.get('overall_quality', 0)
        print(f"   • Overall Quality Score: {overall_quality:.1f}%")
        print(f"   • Completeness: {quality.get('completeness_pct', 0):.1f}%")
        print(f"   • Duplicates: {quality.get('duplicate_rows', 0):,}")
        
        if temporal:
            print(f"\n📅 Temporal Coverage:")
            print(f"   • Time Span: {temporal.get('time_span_days', 'N/A')} days")
            print(f"   • Peak Hour: {temporal.get('peak_hour', 'N/A')}:00")
            print(f"   • Peak Day: {temporal.get('peak_day', 'N/A')}")
        
        print(f"\n🎯 Model Readiness:")
        print(f"   • Status: {model.get('overall_readiness', 'UNKNOWN')}")
        print(f"   • Sample Size: {model.get('sample_size', 'N/A'):,}")
        if 'class_balance_score' in model:
            print(f"   • Fraud Rate: {model.get('class_balance_score', 0):.2f}%")
        
        recommendations = results.get('recommendations', [])
        critical = sum(1 for r in recommendations if r.get('priority') == 'CRITICAL')
        high = sum(1 for r in recommendations if r.get('priority') == 'HIGH')
        
        print(f"\n💡 Recommendations:")
        print(f"   • Total: {len(recommendations)}")
        print(f"   • Critical: {critical}")
        print(f"   • High Priority: {high}")
        
        print("\n" + "=" * 120)
        print("\n✨ ANALYSIS COMPLETE!")
        print(f"\nOutputs generated:")
        print(f"   • Analysis Results: {config.OUTPUT_PICKLE}")
        print(f"   • Log File: {config.LOG_FILE}")
        print("\n" + "=" * 120)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"\nPlease ensure '{config.INPUT_CSV}' exists in the current directory.")
        print(f"Current directory: {Path.cwd()}")
        return 1
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        print("\nCheck the log file for details.")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())