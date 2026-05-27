"""
Experimental design and data processing
"""

from enum import IntEnum
import numpy as np
import pandas as pd


LIGHT_ONSET = {
    "RT": np.array(
        [
            300,
            390,
            440,
            505,
            575,
            635,
            690,
            770,
            840,
            900,
            955,
            1025,
            1115,
            1175,
            1245,
            1310,
            1395,
            1470,
            1535,
            1590,
            1650,
            1720,
            1800,
            1890,
            1960,
            2010,
            2075,
            2165,
            2215,
            2305,
        ],
        dtype=float,
    ),
    "LC": np.array(
        [
            300,
            390,
            480,
            570,
            660,
            750,
            840,
            930,
            1020,
            1110,
            1200,
            1290,
            1380,
            1470,
            1560,
            1650,
            1740,
            1830,
            1920,
            2010,
            2100,
            2190,
            2280,
            2370,
            2460,
            2550,
            2640,
            2730,
            2820,
            2910,
        ],
        dtype=float,
    ),
    "UNP": np.array(
        [
            300,
            390,
            480,
            570,
            660,
            750,
            840,
            930,
            1020,
            1110,
            1200,
            1290,
            1380,
            1470,
            1560,
            1650,
            1740,
            1830,
            1920,
            2010,
            2100,
            2190,
            2280,
            2370,
            2460,
            2550,
            2640,
            2730,
            2820,
            2910,
        ],
        dtype=float,
    ),
}

TONE_ONSET = {
    "RT": np.array([], dtype=float),
    "LC": np.array(
        [
            375,
            495,
            645,
            765,
            930,
            1035,
            1185,
            1320,
            1485,
            1590,
            1725,
            1830,
            1920,
            2085,
            2220,
            2295,
            2400,
            2565,
            2730,
            2895,
        ],
        dtype=float,
    ),
    "UNP": np.array(
        [
            330,
            465,
            585,
            735,
            855,
            1020,
            1125,
            1260,
            1365,
            1440,
            1665,
            1890,
            2010,
            2190,
            2265,
            2370,
            2475,
            2535,
            2655,
            2835,
        ],
        dtype=float,
    ),
}

SHOCK_ONSET = {
    "RT": np.array([], dtype=float),
    "LC": TONE_ONSET["LC"] + 28,
    "UNP": np.array(
        [
            403,
            523,
            673,
            793,
            958,
            1063,
            1213,
            1348,
            1513,
            1618,
            1753,
            1858,
            1948,
            2113,
            2248,
            2323,
            2428,
            2593,
            2758,
            2923,
        ],
        dtype=float,
    ),
}

TRIAL_TYPE = {
    "RT": np.ones_like(LIGHT_ONSET["LC"], dtype=int),
    "LC": np.array(
        [
            1,
            2,
            3,
            1,
            2,
            3,
            1,
            4,
            3,
            1,
            2,
            1,
            5,
            1,
            3,
            1,
            5,
            1,
            2,
            4,
            4,
            1,
            2,
            1,
            5,
            3,
            1,
            5,
            1,
            3,
            1,
            4,
            1,
            2,
        ],
        dtype=int,
    ),
    "UNP": None,
}


rt_lc_unp_state_spec = (
    3,
    4,
    4,
)  # 3 locations, 1 + X light (encode time, L1, ..., LX  each ?sec), 1 + X tone (encode time, T1, ..., TX each ?sec)


def downsample_behavior_data(behavior_data, frequency):
    """
    Downsample behavior data to a specified frequency.

    Args:
    - behavior_data (pd.DataFrame): A Pandas DataFrame containing the behavior data.
    - frequency (str): The frequency to downsample to, in Pandas resample format (e.g. '500ms').

    Returns:
    - behavior_data_ds (pd.DataFrame): A Pandas DataFrame containing the downsampled behavior data.
    """

    list_of_column_names = list(behavior_data.columns)
    behavior_data_ds = pd.DataFrame()

    for i in range(1, len(list_of_column_names)):
        if list_of_column_names[i] == "IN PLATFORM":
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[list_of_column_names[i]]
                .resample(frequency)
                .last()
            )
        elif list_of_column_names[i] == "IN REWARD ZONE":
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[list_of_column_names[i]]
                .resample(frequency)
                .last()
            )
        elif list_of_column_names[i] == "IN CENTER":
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[list_of_column_names[i]]
                .resample(frequency)
                .last()
            )
        elif list_of_column_names[i] == "NEW SPEAKER ACTIVE":
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[list_of_column_names[i]]
                .fillna(0)
                .resample(frequency)
                .last()
            )
        elif list_of_column_names[i] == "SHOCKER ON ACTIVE":
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[list_of_column_names[i]]
                .fillna(0)
                .resample(frequency)
                .last()
            )
        else:
            output = (
                behavior_data.set_index(
                    pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
                )[list_of_column_names[i]]
                .resample(frequency)
                .mean()
            )

        output.bfill(inplace=True)
        output.index = output.index.total_seconds()
        behavior_data_ds[list_of_column_names[i]] = output

    return behavior_data_ds


def process_data(df, phase):
    """Prepare the states
    Transform location, light and tone to state-ready form.
    """
    df = df[["IN PLATFORM", "IN CENTER", "IN REWARD ZONE"]]
    light_onset = pd.DataFrame(
        {"light_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in LIGHT_ONSET[phase]:
        light_onset.loc[t : t + 9, "light_onset"] = 1  # slice includes stop
        light_onset.loc[t + 10 : t + 19, "light_onset"] = 2  # slice includes stop
        light_onset.loc[t + 20 : t + 29, "light_onset"] = 3  # slice includes stop

    tone_onset = pd.DataFrame(
        {"tone_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in TONE_ONSET[phase]:
        tone_onset.loc[t : t + 14, "tone_onset"] = 1
        tone_onset.loc[t + 15 : t + 24, "tone_onset"] = 2
        tone_onset.loc[t + 25 : t + 29, "tone_onset"] = 3

    shock_onset = pd.DataFrame(
        {"shock_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in SHOCK_ONSET[phase]:
        shock_onset.loc[t - 3 : t + 1, "shock_onset"] = 1  # slice includes stop

    features = pd.concat([light_onset, tone_onset, shock_onset], axis=1)

    # Concatenate the original DataFrame and the new DataFrames into a single DataFrame
    df2 = pd.concat(
        [
            df.reset_index(drop=True),
            features.reset_index(drop=True),
        ],
        axis=1,
    )

    df2.set_index(df.index, inplace=True)

    return df2


class StateAxis(IntEnum):
    """State tuple specification
    State is Cartesian product of location, light and tone.
    This enum defines the indices of the state tuple.
    """

    Loc = 0
    Light = 1
    Tone = 2


class Location(IntEnum):
    P = 0
    C = 1
    R = 2


def row_to_state(row):
    """Convert a DataFrame row to a state tuple"""
    s = np.zeros(3, dtype=int)
    if row["IN PLATFORM"] > 0:
        s[StateAxis.Loc] = Location.P
    elif row["IN REWARD ZONE"] > 0:
        s[StateAxis.Loc] = Location.R
    else:
        s[StateAxis.Loc] = Location.C

    s[StateAxis.Light] = row["light_onset"]
    s[StateAxis.Tone] = row["tone_onset"]

    return s
