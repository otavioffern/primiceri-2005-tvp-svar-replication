from __future__ import annotations

from io import BytesIO
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests


FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPCTPI,UNRATE,TB3MS"
VARIABLES = ["inflation", "unemployment", "tbill_3m"]


def fetch_fred_proxy(timeout: int = 30) -> pd.DataFrame:
    """Fetch FRED proxy data for the three-variable Primiceri system.

    The paper uses S&P DRI data. This replication uses public FRED proxies:
    GDPCTPI for the chain-weighted GDP price index, UNRATE for unemployment,
    and TB3MS for the 3-month Treasury bill rate.
    """

    response = requests.get(FRED_GRAPH_URL, timeout=timeout)
    response.raise_for_status()

    if response.content[:2] == b"PK":
        frames = []
        with ZipFile(BytesIO(response.content)) as archive:
            for csv_name in archive.namelist():
                if not csv_name.lower().endswith(".csv"):
                    continue
                with archive.open(csv_name) as handle:
                    frame = pd.read_csv(handle, na_values=".", parse_dates=["observation_date"])
                    frames.append(frame.rename(columns={"observation_date": "date"}).set_index("date"))
        raw = pd.concat(frames, axis=1).sort_index()
    else:
        raw = pd.read_csv(StringIO(response.text), na_values=".", parse_dates=["observation_date"])
        raw = raw.rename(columns={"observation_date": "date"}).set_index("date").sort_index()

    quarterly = pd.DataFrame(index=pd.date_range("1947-01-01", "2001-07-01", freq="QS"))
    quarterly["gdp_price_index"] = raw["GDPCTPI"].resample("QS").last()
    quarterly["unemployment"] = raw["UNRATE"].resample("QS").mean()
    quarterly["tbill_3m"] = raw["TB3MS"].resample("QS").mean()
    quarterly["inflation"] = 400.0 * np.log(
        quarterly["gdp_price_index"] / quarterly["gdp_price_index"].shift(1)
    )

    sample = quarterly.loc["1953-01-01":"2001-07-01", VARIABLES].dropna()
    sample.index.name = "date"
    return sample


def load_or_fetch(output_path: Path, refresh: bool = False) -> pd.DataFrame:
    """Load cached processed data or fetch and cache it."""

    if output_path.exists() and not refresh:
        data = pd.read_csv(output_path, parse_dates=["date"]).set_index("date")
        return data[VARIABLES]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_fred_proxy()
    data.to_csv(output_path, index=True)
    return data
