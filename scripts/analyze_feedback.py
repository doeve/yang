import pandas as pd
import glob
import os
import sys

def analyze_feedback(data_dir="./data/feedback"):
    # Find latest feedback files
    bad_files = glob.glob(os.path.join(data_dir, "bad_decisions_*.csv"))
    missed_files = glob.glob(os.path.join(data_dir, "missed_opportunities_*.csv"))
    
    if not bad_files or not missed_files:
        print("No feedback files found.")
        return

    bad_df = pd.read_csv(max(bad_files, key=os.path.getctime))
    missed_df = pd.read_csv(max(missed_files, key=os.path.getctime))
    
    print("="*60)
    print("FEEDBACK ANALYSIS")
    print("="*60)

    # 1. NO vs YES Performance (Bad Decisions)
    print("\n[Bad Decisions by Side]")
    if not bad_df.empty:
        # Convert pnl to numeric if string
        if bad_df['pnl'].dtype == object:
            bad_df['pnl_val'] = bad_df['pnl'].str.rstrip('%').astype(float)
        else:
            bad_df['pnl_val'] = bad_df['pnl']
            
        side_stats = bad_df.groupby('side').agg({
            'decision_type': 'count',
            'pnl_val': 'mean',
            'confidence': lambda x: x.str.rstrip('%').astype(float).mean() if x.dtype == object else x.mean(),
            'expected_return': lambda x: x.str.rstrip('%').astype(float).mean() if x.dtype == object else x.mean()
        }).rename(columns={'decision_type': 'count', 'pnl_val': 'avg_loss_pct'})
        print(side_stats)
        
        print("\n[Bad Decision Reasons by Side]")
        print(pd.crosstab(bad_df['side'], bad_df['decision_type']))
    else:
        print("No bad decisions data.")

    # 2. NO vs YES Opportunities (Missed)
    print("\n[Missed Opportunities by Type]")
    if not missed_df.empty:
        if missed_df['potential_return_pct'].dtype == object:
            missed_df['potential_return_val'] = missed_df['potential_return_pct'].str.rstrip('%').astype(float)
        else:
            missed_df['potential_return_val'] = missed_df['potential_return_pct']
            
        opp_stats = missed_df.groupby('opportunity_type').agg({
            'tick': 'count',
            'potential_return_val': 'mean',
            'confidence': lambda x: x.str.rstrip('%').astype(float).mean() if x.dtype == object else x.mean(),
            'expected_return': lambda x: x.str.rstrip('%').astype(float).mean() if x.dtype == object else x.mean()
        }).rename(columns={'tick': 'count', 'potential_return_val': 'avg_missed_return_pct'})
        print(opp_stats)
    else:
        print("No missed opportunities data.")

    # 3. Buy vs Hold/Wait Analysis (Inferential)
    # Looking at missed opportunities where confidence was somewhat high but action was WAIT
    print("\n[High Confidence Misses (Frozen / Failed Trigger)]")
    # Convert confidence to float
    if not missed_df.empty:
        missed_df['conf_val'] = missed_df['confidence'].str.rstrip('%').astype(float) if missed_df['confidence'].dtype == object else missed_df['confidence']
        
        # High confidence misses (> 30%)
        high_conf_misses = missed_df[missed_df['conf_val'] > 30]
        print(f"Count of High Confidence (>30%) Misses: {len(high_conf_misses)}")
        print(high_conf_misses[['opportunity_type', 'conf_val', 'expected_return', 'time_remaining']].head(10))
        
        if not high_conf_misses.empty:
            print("\nStats for High Confidence Misses:")
            print(high_conf_misses.groupby('opportunity_type')['conf_val'].describe())

if __name__ == "__main__":
    analyze_feedback()
