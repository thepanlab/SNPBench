import os, re
import numpy as np
import pandas as pd

# Load specified fields from main dataset
def load_main_dataset(path, fids, nrows=None):
    pattern = re.compile(
        r"^(eid|" + "|".join(f"{fid}-" for fid in fids) + r")"
    )
    cols_avail = pd.read_csv(path, nrows=0).columns
    keep_cols = [c for c in cols_avail if pattern.match(c)]
    return pd.read_csv(path, usecols=keep_cols, 
                       nrows=nrows, low_memory=False)

def binarize_field(df, treat_missing_as_zero=False):
    """
    Binarization rules, for each row:
      - returns 1 if any column >= 1
      - returns 0 if all non-missing columns == 0
      - returns np.nan if all columns are missing
    """
    # nullable integer dtype
    df_int = df.astype("Int64")
    # mask: True if any value >=1 in the row
    any_cases = (df_int >= 1).any(axis=1)
    # mask: True if at least one non-missing value exists
    #       and all such values are zero
    all_controls = (df_int.fillna(0).sum(axis=1) == 0) & (df_int.notna().any(axis=1))
    default = 0 if treat_missing_as_zero else np.nan
    result = np.select(
        [any_cases, all_controls],
        [1, 0], 
        default=default
    )
    return pd.Series(result, index=df.index).astype("Int64")

# get first non-missing value per row
def first_field(df):
    return df.bfill(axis=1).iloc[:, 0]

# get last non-missing value per row
def last_field(df):
    return df.ffill(axis=1).iloc[:, -1]

# parse main dataset according to specs (field_name, field_id, field_type)
def parse_main_dataset(main_raw_df, specs):
    fields = []
    print(f"Parsing main dataset...")
    for field_name, field_id, field_type in specs:
        print(f"{field_name} (fid={field_id}, type={field_type})")
        # get current field dataframe
        field_df = main_raw_df.filter(regex=f"^{field_id}-").copy()
        # ensure field_id exists
        if field_df.empty:
            raise ValueError(f"'{field_name}' ({field_id}) not found.")
        # parse the field according to its type
        if field_type.lower() == 'numeric':
            clean_field = last_field(field_df).astype('Float64')
            clean_field = clean_field.rename(field_name)
        elif field_type.lower() == 'binary':
            clean_field = binarize_field(field_df, treat_missing_as_zero=False)
            clean_field = clean_field.rename(field_name)
        elif field_type.lower() == 'raw':
            # return the data as is, but renamed cols as <field_name>_<idx>,
            # if theres just one col, then <field_name>
            clean_field = field_df.copy()
            if clean_field.shape[1] == 1:
                cols = [field_name]
            else:
                cols = [f"{field_name}_{i+1}" 
                        for i in range(clean_field.shape[1])]
            clean_field.columns = cols
        else:
            raise ValueError(f"Invalid field type '{field_type}' for field {field_name} "
                             f"(fid {field_id}). Expected 'numeric' or 'binary'.")

        fields.append(clean_field)
    if 'eid' in main_raw_df.columns:
        return pd.concat([main_raw_df['eid'].astype(str), 
                          pd.concat(fields, axis=1)], axis=1)
    else:
        return pd.concat(fields, axis=1)
