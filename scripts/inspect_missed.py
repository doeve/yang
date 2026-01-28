import pandas as pd
import json

def inspect_missed_q_values():
    df = pd.read_csv("data/feedback/missed_opportunities_20260129_004133.csv")
    
    # Clean confidence column
    if df['confidence'].dtype == object:
        df['conf_val'] = df['confidence'].str.rstrip('%').astype(float)
    else:
        df['conf_val'] = df['confidence']
        
    # Filter high confidence misses
    high_conf = df[df['conf_val'] > 60].head(10)
    
    print("Checking Q-values for top high-confidence misses:")
    
    # We need to find these in the log file to get the Q-values (CSV doesn't have them in list format well)
    # Actually, the CSV *doesn't* have q-values column populated in my previous script? 
    # Let's check the JSON which I saved!
    
    with open("data/feedback/missed_opportunities_20260129_004133.json", 'r') as f:
        data = json.load(f)
        
    # Filter in python
    misses = [d for d in data if d['confidence'] > 0.60]
    
    print(f"Found {len(misses)} high confidence misses.")
    
    for i, m in enumerate(misses[:10]):
        q = m['q_values']
        # Actions: 0=WAIT, 1=BUY_YES, 2=BUY_NO, 3=EXIT, 4=HOLD
        max_q = max(q)
        max_act = q.index(max_q)
        
        print(f"\nMiss #{i+1}: Type={m['opportunity_type']}, Conf={m['confidence']:.2%}, Return={m['expected_return']:.2%}")
        print(f"  Q-values: {q}")
        print(f"  Max Q Action: {max_act} (Value: {max_q:.2f})")
        print(f"  Q(BUY_YES): {q[1]:.2f}")
        print(f"  Q(BUY_NO):  {q[2]:.2f}")
        
        # Check gap
        if m['opportunity_type'] == 'YES':
            gap = q[0] - q[1]
            print(f"  Gap to BUY_YES: {gap:.2f}")
        else:
            gap = q[0] - q[2]
            print(f"  Gap to BUY_NO: {gap:.2f}")

if __name__ == "__main__":
    inspect_missed_q_values()
