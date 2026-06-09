"""Load real CFPB consumer complaint data bundled with the app."""

import pathlib
import pandas as pd

_DATA_PATH = pathlib.Path(__file__).parent / "cfpb_sample.csv"


def load_sample(n: int = 500) -> pd.DataFrame:
    """
    Return up to n real CFPB complaints from the bundled sample file.
    Columns: complaint_id, date_submitted, product, company, state,
             consumer_narrative, outcome.
    """
    df = pd.read_csv(_DATA_PATH)
    if n < len(df):
        df = df.sample(n, random_state=42)
    return df.reset_index(drop=True)
