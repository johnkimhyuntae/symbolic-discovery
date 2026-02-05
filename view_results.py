#!/usr/bin/env python3
"""
Quick results viewer for runner.py output.
Displays a clean summary table of experiment results.
"""

import argparse
import pandas as pd
from pathlib import Path

def view_results(csv_path: str, mode: str = 'summary'):
    """
    View experiment results in different formats.
    
    Args:
        csv_path: Path to results CSV file
        mode: Display mode - 'summary', 'full', 'compare', 'stats', 'failures', or 'interesting'
    """
    if not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    if mode == 'summary':
        # Clean, compact summary
        print(f"\n{'='*80}")
        print(f"📊 EXPERIMENT RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        for _, row in df.iterrows():
            status_icon = "✓" if row['status'] == "Success" else "✗"
            eq_clean = row['found_eq'][:50] + "..." if len(row['found_eq']) > 50 else row['found_eq']
            
            print(f"{status_icon} [{row['method']:6s}] {row['dataset']:4s} (N={row['noise']:.2f}) "
                  f"→ {eq_clean:40s} R²={row['r2']:.4f} ({row['time_s']:.2f}s)")
        
        print(f"\n{'='*80}")
        print(f"Total runs: {len(df)} | Success: {(df['status']=='Success').sum()} | "
              f"Failed: {(df['status']!='Success').sum()}")
        print(f"{'='*80}\n")
    
    elif mode == 'full':
        # Full details with equations
        print(f"\n{'='*80}")
        print(f"📋 FULL RESULTS")
        print(f"{'='*80}\n")
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        print(df.to_string(index=False))
        print()
    
    elif mode == 'compare':
        # Side-by-side comparison of methods on same datasets
        print(f"\n{'='*80}")
        print(f"⚖️  METHOD COMPARISON")
        print(f"{'='*80}\n")
        
        if 'dataset' in df.columns and 'method' in df.columns:
            pivot = df.pivot_table(
                index=['dataset', 'noise'], 
                columns='method', 
                values='status',
                aggfunc=lambda x: '✓' if (x == 'Success').all() else '✗'
            )
            print(pivot)
            print()
            
            # R² comparison
            print("\nR² Scores:")
            pivot_r2 = df.pivot_table(
                index=['dataset', 'noise'], 
                columns='method', 
                values='r2',
                aggfunc='mean'
            )
            print(pivot_r2.to_string(float_format='%.4f'))
            print()
    
    elif mode == 'stats':
        # Statistical summary
        print(f"\n{'='*80}")
        print(f"📈 STATISTICAL SUMMARY")
        print(f"{'='*80}\n")
        
        print("Success Rate by Method:")
        success_rate = df.groupby('method').apply(
            lambda x: (x['status'] == 'Success').sum() / len(x) * 100
        )
        for method, rate in success_rate.items():
            print(f"  {method:8s}: {rate:5.1f}%")
        
        print("\nAverage R² by Method (successful runs only):")
        avg_r2 = df[df['status'] == 'Success'].groupby('method')['r2'].mean()
        for method, r2 in avg_r2.items():
            print(f"  {method:8s}: {r2:.4f}")
        
        print("\nAverage Runtime by Method:")
        avg_time = df.groupby('method')['time_s'].mean()
        for method, time in avg_time.items():
            print(f"  {method:8s}: {time:.3f}s")
        
        print("\nSuccess Rate by Noise Level:")
        if 'noise' in df.columns:
            noise_success = df.groupby('noise').apply(
                lambda x: (x['status'] == 'Success').sum() / len(x) * 100
            )
            for noise, rate in noise_success.items():
                print(f"  {noise:.2f}: {rate:5.1f}%")
        
        print()

    elif mode == 'failures':
        # Only failures/errors, with full equations
        print(f"\n{'='*80}")
        print(f"✗ FAILURES / ERRORS")
        print(f"{'='*80}\n")

        failed = df[df['status'] != 'Success'].copy()
        if failed.empty:
            print("(none)")
            return

        cols = [c for c in ['run_id', 'dataset', 'method', 'noise', 'seed', 'found_eq', 'r2', 'time_s', 'status'] if c in failed.columns]
        failed = failed.sort_values(['dataset', 'method', 'noise', 'seed'])
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        print(failed[cols].to_string(index=False))
        print(f"\nTotal failures: {len(failed)}")

    elif mode == 'interesting':
        # Failures + low-R² successes (default threshold: < 0.99)
        print(f"\n{'='*80}")
        print(f"🔎 INTERESTING RUNS (failures + low-R² successes)")
        print(f"{'='*80}\n")

        df_local = df.copy()
        df_local['r2_num'] = pd.to_numeric(df_local.get('r2'), errors='coerce')

        failed = df_local[df_local['status'] != 'Success']
        low_r2 = df_local[(df_local['status'] == 'Success') & (df_local['r2_num'] < 0.99)]

        interesting = pd.concat([failed, low_r2], ignore_index=True)
        if interesting.empty:
            print("(none)")
            return

        cols = [c for c in ['run_id', 'dataset', 'method', 'noise', 'seed', 'found_eq', 'r2', 'time_s', 'status'] if c in interesting.columns]
        interesting = interesting.sort_values(['status', 'dataset', 'method', 'noise', 'seed'])
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.width', None)
        print(interesting[cols].to_string(index=False))
        print(f"\nFailures: {len(failed)} | Low-R² successes: {len(low_r2)}")

def main():
    parser = argparse.ArgumentParser(
        description="View experiment results from runner.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_results.py results.csv                  # Summary view
  python view_results.py results.csv --mode full      # Full details
  python view_results.py results.csv --mode compare   # Compare methods
  python view_results.py results.csv --mode stats     # Statistical summary
    python view_results.py results.csv --mode failures  # Show failures/errors only
    python view_results.py results.csv --mode interesting  # Failures + low-R² successes
        """
    )
    
    parser.add_argument('csv_file', help='Path to results CSV file')
    parser.add_argument('--mode', '-m', 
                       choices=['summary', 'full', 'compare', 'stats', 'failures', 'interesting'],
                       default='summary',
                       help='Display mode (default: summary)')
    
    args = parser.parse_args()
    view_results(args.csv_file, args.mode)

if __name__ == "__main__":
    main()
