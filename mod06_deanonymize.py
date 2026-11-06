import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    
    keys = ["age", "gender", "zip3"]

    # Find unique records in each dataset
    anon_unique = anon_df[
        ~anon_df.duplicated(subset=keys, keep=False)
    ]

    aux_unique = aux_df[
        ~aux_df.duplicated(subset=keys, keep=False)
    ]

    # Match records using the three shared columns
    merged = pd.merge(
        anon_unique,
        aux_unique,
        on=keys,
        how="inner"
    )

    # Return the ID and matching name
    matches = merged[["anon_id", "name"]].copy()
    matches = matches.rename(columns={"name": "matched_name"})

    return matches



def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    
    if len(anon_df) == 0:
        return 0.0

    return len(matches_df) / len(anon_df)

