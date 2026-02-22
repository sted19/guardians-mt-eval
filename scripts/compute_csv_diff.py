import argparse
import re

import numpy as np
import pandas as pd


def normalize_csv1(s: str) -> str:
    """Transform csv1 mt strings to match csv2 format."""
    # The csv1 data contains actual newline characters.
    # csv2 has the literal two-char sequence backslash+n with spaces around it.
    # 1. Replace actual newlines with the literal two-char escape sequence
    s = s.replace("\n", "NEWLINE_PLACEHOLDER")
    # 2. Ensure exactly one space before and after each placeholder
    s = re.sub(r"\s*NEWLINE_PLACEHOLDER\s*", r" \\n ", s)
    # 3. Collapse consecutive escaped newlines into a single one
    s = re.sub(r"(\\n\s*)+\\n", r"\\n", s)
    # 4. Re-enforce exactly one space before and after each escaped newline
    s = re.sub(r"\s*\\n\s*", r" \\n ", s)
    # 5. Strip leading escaped newlines (and surrounding spaces)
    s = re.sub(r"^(\s*\\n\s*)+", "", s)
    # 6. Strip trailing escaped newlines (and surrounding spaces)
    s = re.sub(r"(\s*\\n\s*)+$", "", s)
    s = s.strip()
    return s


def main():
    parser = argparse.ArgumentParser(description="Compute diff between two CSV files.")
    parser.add_argument("csv1", help="Path to the first CSV file.")
    parser.add_argument("csv2", help="Path to the second CSV file.")
    parser.add_argument("-n", "--num-diff-rows", type=int, default=10, help="Number of differing rows to display.")
    args = parser.parse_args()

    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    assert list(df1.columns) == list(df2.columns), (
        f"Headers differ!\n  csv1: {list(df1.columns)}\n  csv2: {list(df2.columns)}"
    )

    # Drop "annotators" column — those differ by default
    df1 = df1.drop(columns=["annotators"], errors="ignore")
    df2 = df2.drop(columns=["annotators"], errors="ignore")

    # Normalize dtypes: cast all columns to string for comparison,
    # except the last column which is compared numerically (z-normalized floats).
    cols = list(df1.columns)
    last_col = cols[-1]
    other_cols = cols[:-1]

    # Cast non-numeric columns to str so type mismatches don't cause false diffs
    # fillna first so NaN == NaN (NaN != NaN by IEEE 754)
    for col in other_cols:
        df1[col] = df1[col].fillna("__NAN__").astype(str)
        df2[col] = df2[col].fillna("__NAN__").astype(str)
    # Normalize csv2's mt column to match csv1's format
    if 'src' in other_cols:
        df1['src'] = df1['src'].map(normalize_csv1)
    if "mt" in other_cols:
        df1["mt"] = df1["mt"].map(normalize_csv1)
    df1[last_col] = df1[last_col].astype(float)
    df2[last_col] = df2[last_col].astype(float)

    # Sort both dataframes by all columns so rows align
    df1_sorted = df1.sort_values(by=cols).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=cols).reset_index(drop=True)

    # Find rows that differ
    # Use exact equality for string columns, np.isclose for the last numeric column.
    exact_diff = (df1_sorted[other_cols] != df2_sorted[other_cols]).any(axis=1)
    approx_diff = ~np.isclose(
        df1_sorted[last_col],
        df2_sorted[last_col],
        atol=1e-6,
        rtol=1e-6,
    )
    diff_mask = exact_diff | approx_diff
    diff_indices = diff_mask[diff_mask].index.tolist()

    n_diff = len(diff_indices)
    print(f"Total rows: {len(df1)}")
    print(f"Differing rows: {n_diff}")

    if n_diff == 0:
        print("The two CSVs contain the same rows.")
        return

    print(f"\nShowing first {min(args.num_diff_rows, n_diff)} differing rows:\n")
    for i, idx in enumerate(diff_indices[:args.num_diff_rows]):
        row1 = df1_sorted.iloc[idx]
        row2 = df2_sorted.iloc[idx]
        print(f"=== Row {idx} ===")
        for col in cols:
            v1, v2 = row1[col], row2[col]
            marker = "  " if v1 == v2 else "!!"
            print(f"  {marker} {col}: {v1!r} vs {v2!r}")
        print()


if __name__ == "__main__":
    main()
