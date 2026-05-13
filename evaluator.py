import pandas as pd

def benchmark_pipeline(extracted_path, ground_truth_path):
    print("Loading datasets for evaluation...\n")
    try:
        # Load the CSVs
        df_ext = pd.read_csv(extracted_path)
        df_truth = pd.read_csv(ground_truth_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both redcap_import_ready.csv and redcap_expected.csv exist in this folder.")
        return

    # Find columns to compare (ignoring record_id so it doesn't crash on '1' vs 'real-001')
    common_cols = df_truth.columns.intersection(df_ext.columns).drop('record_id', errors='ignore')
    missing_cols = df_truth.columns.difference(df_ext.columns)

    if not missing_cols.empty:
        print(f"WARNING: Your extraction pipeline is missing {len(missing_cols)} expected columns:")
        print(f"{list(missing_cols)}\n")

    accuracy_data = []
    total_cells = 0
    correct_cells = 0

    print("=== ERROR LOG (MISMATCHES) ===")

    # Evaluate field-by-field
    for col in common_cols:
        # Safely convert to string and strip decimals/blanks to avoid fake '1' vs '1.0' failures
        ext_col = df_ext[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().replace('nan', '')
        truth_col = df_truth[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip().replace('nan', '')

        # Calculate matches row by row to support batch processing
        total = min(len(ext_col), len(truth_col))
        if total == 0:
            continue

        matches = 0
        for i in range(total):
            if ext_col[i] == truth_col[i]:
                matches += 1
            else:
                # Log the exact argument
                print(f"Row {i+1} - Field '{col}': AI extracted [{ext_col[i]}] | Ground Truth expected [{truth_col[i]}]")

        accuracy = (matches / total) * 100
        accuracy_data.append({"REDCap Field": col, "Accuracy": accuracy})

        total_cells += total
        correct_cells += matches

    # Format the results into a clean table
    if not accuracy_data:
        print("Error: No common fields to evaluate.")
        return

    results_df = pd.DataFrame(accuracy_data)
    results_df = results_df.sort_values(by="Accuracy", ascending=True) # Show weakest fields at the top
    results_df['Accuracy'] = results_df['Accuracy'].round(1).astype(str) + '%'

    print("\n=== FIELD-LEVEL ACCURACY REPORT ===")
    print(results_df.to_string(index=False))

    overall_accuracy = (correct_cells / total_cells) * 100
    print(f"\n===================================")
    print(f"OVERALL PIPELINE ACCURACY: {overall_accuracy:.1f}%")
    print(f"===================================")

if __name__ == "__main__":
    # The output from your batch_redcap_extractor.py
    EXTRACTED_FILE = 'redcap_import_ready.csv'

    # The answer key Stephen mentioned
    GROUND_TRUTH_FILE = 'redcap_expected.csv'

    benchmark_pipeline(EXTRACTED_FILE, GROUND_TRUTH_FILE)