"""
Main entry point for the Dormancy Fraud Detection EDA Pipeline.

Run from the project root (or src/ directory):
    python src/Main.py

Checkpointing
-------------
Completed sections are saved to eda_checkpoints/<section>.joblib.
On re-run, those sections are loaded from disk and skipped — only
incomplete or missing sections are computed.

To force a full re-run:       delete the eda_checkpoints/ folder.
To re-run a single section:   delete eda_checkpoints/<section>.joblib.
"""

import sys
import io

# Force UTF-8 on stdout so emoji/special chars don't crash on Windows cp1252 terminals.
# Must be done before any print() calls or imports that print.
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )

from pathlib import Path

from Config import EDAConfig
from Fraud_eda import FraudEDA


def print_header(config: EDAConfig):
    width = 120
    print("=" * width)
    print(" " * 35 + "FINANCIAL FRAUD DETECTION EDA")
    print(" " * 40 + "Production Version 3.0")
    print("=" * width)
    print(f"\n📋 Configuration:")
    print(f"   Input File          : {config.INPUT_CSV}")
    print(f"   Output File         : {config.OUTPUT_PICKLE}")
    print(f"   Log File            : {config.LOG_FILE}")
    print(f"   Checkpoint Dir      : {config.CHECKPOINT_DIR}/")
    print(f"   ML Sample Size      : {config.SAMPLE_SIZE:,} rows (stratified)")
    print(f"   Silhouette Sample   : {config.SILHOUETTE_SAMPLE:,} rows")
    print(f"   Correlation Threshold: {config.CORRELATION_THRESHOLD}")
    print(f"   PCA Variance Threshold: {config.PCA_VARIANCE_THRESHOLD}")
    print(f"   Random State        : {config.RANDOM_STATE}")
    print("=" * width)


def print_summary(results: dict, config: EDAConfig):
    width = 120
    meta    = results.get('metadata', {})
    quality = results.get('data_quality', {})
    model   = results.get('model_readiness', {})
    recs    = results.get('recommendations', [])

    n_critical = sum(1 for r in recs if r['priority'] == 'CRITICAL')
    n_high     = sum(1 for r in recs if r['priority'] == 'HIGH')
    n_medium   = sum(1 for r in recs if r['priority'] == 'MEDIUM')

    temporal = results.get('temporal_analysis', {})
    feature  = results.get('feature_importance', {})

    print("\n" + "=" * width)
    print("ANALYSIS SUMMARY")
    print("=" * width)

    print(f"\n📊 Dataset Information:")
    print(f"   • Total Transactions : {meta.get('total_rows', 'N/A'):,}")
    print(f"   • Total Features     : {meta.get('total_columns', 'N/A')}")
    print(f"   • Memory Usage       : {meta.get('memory_mb', 0):.2f} MB")

    print(f"\n✅ Data Quality:")
    print(f"   • Overall Quality Score : {quality.get('overall_quality', 0):.1f}%")
    print(f"   • Completeness          : {quality.get('completeness_pct', 0):.1f}%")
    print(f"   • Duplicates            : {quality.get('duplicate_rows', 0)}")

    if temporal:
        print(f"\n📅 Temporal Coverage:")
        print(f"   • Time Span  : {temporal.get('time_span_days', 'N/A')} days")
        print(f"   • Peak Hour  : {temporal.get('peak_hour', 'N/A')}:00")
        print(f"   • Peak Day   : {temporal.get('peak_day', 'N/A')}")

    if feature.get('rf_importance'):
        top5 = sorted(
            feature['rf_importance'].items(), key=lambda x: x[1], reverse=True
        )[:5]
        print(f"\n🏆 Top 5 Features (Random Forest):")
        for feat, imp in top5:
            print(f"   • {feat}: {imp:.4f}")

    print(f"\n🎯 Model Readiness:")
    print(f"   • Status      : {model.get('overall_readiness', 'UNKNOWN')}")
    print(f"   • Sample Size : {model.get('sample_size', 0):,}")
    print(f"   • Fraud Rate  : {model.get('class_balance_score', 0):.2f}%")
    print(f"   • Features    : {model.get('feature_count', 0)} (leak-free)")

    leaky_found = model.get('leaky_columns_excluded', [])
    if leaky_found:
        print(f"   ⚠️  Leaky columns still present: {leaky_found}")
    else:
        print(f"   ✅ No leaky columns in dataframe")

    print(f"\n💡 Recommendations:")
    print(f"   • Total    : {len(recs)}")
    print(f"   • Critical : {n_critical}")
    print(f"   • High     : {n_high}")
    print(f"   • Medium   : {n_medium}")

    print(f"\n{'=' * width}")
    print(f"\n✨ ANALYSIS COMPLETE!\n")
    print(f"Outputs generated:")
    print(f"   • Analysis Results : {config.OUTPUT_PICKLE}")
    print(f"   • Checkpoints      : eda_checkpoints/")
    print(f"   • Log File         : eda_analysis.log")
    print(f"\n{'=' * width}\n")


def main():
    config = EDAConfig()
    print_header(config)

    try:
        eda     = FraudEDA(config)
        results = eda.run_full_analysis()
        eda.save_results()
        print_summary(results, config)
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("   Check that INPUT_CSV in Config.py points to the correct path.")
        return 1

    except MemoryError:
        print("\n❌ Out of memory.")
        print("   Reduce SAMPLE_SIZE or SILHOUETTE_SAMPLE in Config.py.")
        return 1

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())