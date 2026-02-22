"""Concatenate two CSV files with safety checks on columns and dtypes."""

from argparse import ArgumentParser
import sys

import pandas as pd


def parse_args():
    parser = ArgumentParser(description="Safely concatenate two CSV files with matching schemas.")
    parser.add_argument("file_a", type=str, help="Path to the first CSV file.")
    parser.add_argument("file_b", type=str, help="Path to the second CSV file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument(
        "--allow-missing-cols",
        action="store_true",
        help="If set, allow missing columns by filling them with NaN instead of raising an error.",
    )
    return parser.parse_args()


def validate_and_concat(df_a: pd.DataFrame, df_b: pd.DataFrame, allow_missing_cols: bool) -> pd.DataFrame:
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)

    only_in_a = cols_a - cols_b
    only_in_b = cols_b - cols_a

    if only_in_a or only_in_b:
        msg = "Column mismatch between the two files:\n"
        if only_in_a:
            msg += f"  Only in file A: {sorted(only_in_a)}\n"
        if only_in_b:
            msg += f"  Only in file B: {sorted(only_in_b)}\n"

        if not allow_missing_cols:
            raise ValueError(msg + "Use --allow-missing-cols to concatenate anyway (missing columns will be NaN).")
        else:
            print(f"WARNING: {msg}Missing columns will be filled with NaN.", file=sys.stderr)

    # Check dtypes on shared columns.
    # When a column is entirely NaN, pandas infers float64 instead of the
    # "real" dtype. We treat such mismatches as benign and coerce the
    # all-NaN column to the other file's dtype.
    shared_cols = sorted(cols_a & cols_b)
    dtype_mismatches = []
    for col in shared_cols:
        dt_a, dt_b = df_a[col].dtype, df_b[col].dtype
        if dt_a == dt_b:
            continue

        a_all_nan = df_a[col].isna().all()
        b_all_nan = df_b[col].isna().all()

        if a_all_nan and not b_all_nan:
            # File A's column is all NaN → adopt file B's dtype
            df_a[col] = df_a[col].astype(dt_b)
            print(f"INFO: Column '{col}' is all-NaN in file A, cast to {dt_b} (from file B).", file=sys.stderr)
        elif b_all_nan and not a_all_nan:
            df_b[col] = df_b[col].astype(dt_a)
            print(f"INFO: Column '{col}' is all-NaN in file B, cast to {dt_a} (from file A).", file=sys.stderr)
        elif a_all_nan and b_all_nan:
            # Both all-NaN, just unify to the same dtype
            df_b[col] = df_b[col].astype(dt_a)
        else:
            dtype_mismatches.append((col, dt_a, dt_b))

    if dtype_mismatches:
        msg = "Dtype mismatches on shared columns:\n"
        for col, dt_a, dt_b in dtype_mismatches:
            msg += f"  '{col}': file A has {dt_a}, file B has {dt_b}\n"

        all_scores_a = df_a["score"].tolist() if "score" in df_a.columns else []
        all_scores_b = df_b["score"].tolist() if "score" in df_b.columns else []

        all_non_nan_scores_a = [s for s in all_scores_a if pd.notna(s)]
        all_non_nan_scores_b = [s for s in all_scores_b if pd.notna(s)]

        nan_scores_idx_b = [i for i, s in enumerate(all_scores_b) if pd.isna(s)]

        raise ValueError(msg + "Please ensure both files have the same dtypes before concatenating.")

    # Check column order (warn only)
    if list(df_a.columns) != list(df_b.columns) and not (only_in_a or only_in_b):
        print(
            "WARNING: Columns are in different order. Output will use column order from file A.",
            file=sys.stderr,
        )

    # Check for duplicate rows
    combined = pd.concat([df_a, df_b], ignore_index=True)
    n_dupes = combined.duplicated().sum()
    if n_dupes > 0:
        print(f"WARNING: {n_dupes} duplicate row(s) found after concatenation.", file=sys.stderr)

    return combined


def main():
    args = parse_args()

    df_a = pd.read_csv(args.file_a)
    df_b = pd.read_csv(args.file_b)

    print(f"File A: {args.file_a}  ({len(df_a)} rows, {len(df_a.columns)} cols)")
    print(f"File B: {args.file_b}  ({len(df_b)} rows, {len(df_b.columns)} cols)")

    combined = validate_and_concat(df_a, df_b, allow_missing_cols=args.allow_missing_cols)

    combined.to_csv(args.output, index=False)
    print(f"Output: {args.output}  ({len(combined)} rows, {len(combined.columns)} cols)")


if __name__ == "__main__":
    main()
