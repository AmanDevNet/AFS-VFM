import pandas as pd
import os

def merge_csvs():
    results_dir = os.path.join("D:\\", "Research Project", "results")
    
    files = ["batch_1.csv", "batch_2.csv", "batch_3.csv"]
    dataframes = []
    
    for f in files:
        filepath = os.path.join(results_dir, f)
        if not os.path.exists(filepath):
            print(f"Error: {f} not found in {results_dir}")
            return
        
        print(f"Loading {f} ...")
        df = pd.read_csv(filepath)
        print(f"  -> {len(df)} rows")
        dataframes.append(df)
        
    print("\nMerging data ...")
    combined = pd.concat(dataframes, ignore_index=True)
    
    output_path = os.path.join(results_dir, "full_benchmark.csv")
    combined.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print(f" MERGE COMPLETE")
    print(f" Total rows: {len(combined)}")
    print(f" Saved to:   {output_path}")
    print("="*50)

if __name__ == "__main__":
    merge_csvs()
