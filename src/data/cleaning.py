def drop_missing_by_row(df, verbose=True):
    before = df.shape
    after = df.shape
    rows_lost = before[0] - after[0]
    pct_lost = 100 * (1 - after[0] / before[0])
    df = df.dropna(axis=0)
    if verbose:
        print("Dropping missing values by row")
        print(f"  shape before row drop {before}")
        print(f"  shape after row drop {after}")
        print(f"  rows lost = {rows_lost}, {pct_lost:.2f} %")
        print("\n")
    return df

