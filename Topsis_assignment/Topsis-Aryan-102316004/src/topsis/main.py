import sys
import pandas as pd
import numpy as np
import os

def check_inputs():
    if len(sys.argv) != 5:
        print("Usage: python <script.py> <InputDataFile> <Weights> <Impacts> <OutputResultFileName>")
        sys.exit(1)

    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    output_file = sys.argv[4]

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    return input_file, weights, impacts, output_file

def validate_and_load_data(input_file):
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    if df.shape[1] < 3:
        print("Error: Input file must contain three or more columns.")
        sys.exit(1)

    # Check if columns from 2nd to last are numeric
    numeric_df = df.iloc[:, 1:]
    
    if not numeric_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all():
         print("Error: From 2nd to last columns must contain numeric values only.")
         sys.exit(1)

    return df

def topsis_logic(df, weights, impacts, output_file):
    # Matrix of alternatives (m) x criteria (n)
    matrix = df.iloc[:, 1:].values.astype(float)
    rows, cols = matrix.shape

    try:
        weight_list = [float(w) for w in weights.split(',')]
        impact_list = impacts.split(',')
    except ValueError:
        print("Error: Weights must be numeric and separated by commas.")
        sys.exit(1)
        
    if len(weight_list) != cols or len(impact_list) != cols:
        print("Error: Number of weights, impacts and number of columns (from 2nd to last) must be the same.")
        sys.exit(1)

    if not all(i in ['+', '-'] for i in impact_list):
        print("Error: Impacts must be either +ve or -ve.")
        sys.exit(1)

    # 1. Vector Normalization
    rss = np.sqrt(np.sum(matrix**2, axis=0))
    normalized_matrix = matrix / rss

    # 2. Weighted Normalization
    weighted_matrix = normalized_matrix * weight_list

    # 3. Ideal Best and Ideal Worst
    ideal_best = []
    ideal_worst = []

    for i in range(cols):
        if impact_list[i] == '+':
            ideal_best.append(np.max(weighted_matrix[:, i]))
            ideal_worst.append(np.min(weighted_matrix[:, i]))
        else:
            ideal_best.append(np.min(weighted_matrix[:, i]))
            ideal_worst.append(np.max(weighted_matrix[:, i]))
            
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # 4. Euclidean Distance
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # 5. Performance Score
    score = dist_worst / (dist_best + dist_worst)

    # 6. Rank
    df['Topsis Score'] = score
    df['Rank'] = df['Topsis Score'].rank(ascending=False, method='min').astype(int)

    df.to_csv(output_file, index=False)
    print(f"Result saved to {output_file}")

def main():
    if len(sys.argv) != 5:
        print("Usage: topsis <InputDataFile> <Weights> <Impacts> <OutputResultFileName>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    output_file = sys.argv[4]

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
        
    df = validate_and_load_data(input_file)
    topsis_logic(df, weights, impacts, output_file)

if __name__ == "__main__":
    main()
