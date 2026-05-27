"""
Threat-conditioning task specification.

This module defines the experimental design and data processing for a
threat-conditioning paradigm in which a rodent navigates between three
locations — Platform (P), Center (C), and Reward zone (R) — while being
exposed to combinations of a light cue, a tone cue, and a foot shock.

Three experimental phases are supported:

- ``"RT"``  — reward-only baseline: light cues, no tone/shock
- ``"LC"``  — low-conflict threat conditioning: light and tone co-occur in
              various temporal arrangements; shock follows the tone by ~28 s
- ``"UNP"`` — unpredictable threat: tone and shock timing are not locked
              to light onset

Typical usage
-------------
Load and downsample raw tracking data, then encode composite states::

    import pandas as pd
    from spec import (
        downsample_behavior_data, process_data,
        row_to_state, rt_lc_unp_state_spec,
    )

    df = pd.read_csv("M1.csv", encoding="unicode_escape", header=0)
    df.columns = map(str.upper, df.columns)
    df = downsample_behavior_data(df, "1s")
    df = process_data(df, "LC")

    state = row_to_state(df.iloc[0])   # length-3 integer vector

The composite state vector has shape ``(3,)`` indexed by ``StateAxis``:
``[location, light_level, tone_level]``.  It maps directly to the leading
axes of the SARSA Q-table whose shape is ``(*rt_lc_unp_state_spec, n_actions)``.
"""

from enum import IntEnum
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stimulus timing tables
# ---------------------------------------------------------------------------
# All times are in seconds from session start.  Each array contains the
# onset time of each stimulus presentation for the named phase.

LIGHT_ONSET = {
    # Reward-only baseline: 30 light presentations, irregular spacing.
    "RT": np.array(
        [
            300, 390, 440, 505, 575, 635, 690, 770, 840, 900,
            955, 1025, 1115, 1175, 1245, 1310, 1395, 1470, 1535, 1590,
            1650, 1720, 1800, 1890, 1960, 2010, 2075, 2165, 2215, 2305,
        ],
        dtype=float,
    ),
    # Low-conflict conditioning: 30 light presentations, regular 90-s spacing.
    "LC": np.array(
        [
            300, 390, 480, 570, 660, 750, 840, 930, 1020, 1110,
            1200, 1290, 1380, 1470, 1560, 1650, 1740, 1830, 1920, 2010,
            2100, 2190, 2280, 2370, 2460, 2550, 2640, 2730, 2820, 2910,
        ],
        dtype=float,
    ),
    # Unpredictable threat: same light schedule as LC.
    "UNP": np.array(
        [
            300, 390, 480, 570, 660, 750, 840, 930, 1020, 1110,
            1200, 1290, 1380, 1470, 1560, 1650, 1740, 1830, 1920, 2010,
            2100, 2190, 2280, 2370, 2460, 2550, 2640, 2730, 2820, 2910,
        ],
        dtype=float,
    ),
}

TONE_ONSET = {
    # No tone in the reward-only baseline.
    "RT": np.array([], dtype=float),
    # Low-conflict conditioning: 20 tone presentations.
    # Tones occur at 0 s, ±15 s, or +30 s relative to a light onset,
    # producing the five trial types encoded in TRIAL_TYPE["LC"].
    "LC": np.array(
        [
            375, 495, 645, 765, 930, 1035, 1185, 1320, 1485, 1590,
            1725, 1830, 1920, 2085, 2220, 2295, 2400, 2565, 2730, 2895,
        ],
        dtype=float,
    ),
    # Unpredictable threat: 20 tone presentations with irregular offsets.
    "UNP": np.array(
        [
            330, 465, 585, 735, 855, 1020, 1125, 1260, 1365, 1440,
            1665, 1890, 2010, 2190, 2265, 2370, 2475, 2535, 2655, 2835,
        ],
        dtype=float,
    ),
}

SHOCK_ONSET = {
    # No shock in the reward-only baseline.
    "RT": np.array([], dtype=float),
    # LC: shock is delivered 28 s after each tone onset (during the
    # level-3 tone period), so the animal can avoid by reaching the platform.
    "LC": TONE_ONSET["LC"] + 28,
    # UNP: shock timing is unpredictable (not locked to tone onset).
    "UNP": np.array(
        [
            403, 523, 673, 793, 958, 1063, 1213, 1348, 1513, 1618,
            1753, 1858, 1948, 2113, 2248, 2323, 2428, 2593, 2758, 2923,
        ],
        dtype=float,
    ),
}

# ---------------------------------------------------------------------------
# Trial type labels (LC phase only)
# ---------------------------------------------------------------------------
# TRIAL_TYPE["LC"] has 34 entries — one per trial, ordered by anchor time.
# Trials are either light-anchored (30) or tone-only (4), sorted together.
#
# Mapping:
#   1 — light-only       : light with no nearby tone
#   2 — tone-then-light  : tone starts ~15 s before the light
#   3 — light-then-tone  : tone starts ~15 s after the light
#   4 — copresented      : light and tone start simultaneously (0 s offset)
#   5 — tone-only        : tone with no associated light (~30 s from any light)
TRIAL_TYPE = {
    # RT has only one trial type (light-only); reuse LC light count for shape.
    "RT": np.ones_like(LIGHT_ONSET["LC"], dtype=int),
    "LC": np.array(
        [
            1, 2, 3, 1, 2, 3, 1, 4, 3, 1,
            2, 1, 5, 1, 3, 1, 5, 1, 2, 4,
            4, 1, 2, 1, 5, 3, 1, 5, 1, 3,
            1, 4, 1, 2,
        ],
        dtype=int,
    ),
    # UNP trial types not defined.
    "UNP": None,
}

# ---------------------------------------------------------------------------
# State space specification
# ---------------------------------------------------------------------------
# The composite state is a length-3 integer vector:
#   [location, light_level, tone_level]
#
# Dimensions:
#   location   : 3 values  (P=0, C=1, R=2)
#   light_level: 4 values  (0=off, 1=early, 2=mid, 3=late; each ~10 s)
#   tone_level : 4 values  (0=off, 1=early, 2=mid, 3=late; each ~10 s)
#
# The Q-table shape is (*rt_lc_unp_state_spec, n_actions) = (3, 4, 4, n_actions).
rt_lc_unp_state_spec = (3, 4, 4)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def downsample_behavior_data(behavior_data, frequency):
    """Downsample raw tracking data to a uniform time grid.

    The raw CSV is sampled at sub-second resolution with variable spacing.
    This function resamples every column to a fixed ``frequency`` so that
    the resulting DataFrame has a regular integer-second index suitable
    for state encoding.

    Binary location columns (``IN PLATFORM``, ``IN REWARD ZONE``,
    ``IN CENTER``) and event columns (``NEW SPEAKER ACTIVE``,
    ``SHOCKER ON ACTIVE``) are resampled with ``last()`` to preserve
    instantaneous values.  All other columns (e.g. speed, coordinates)
    are averaged with ``mean()``.

    Parameters
    ----------
    behavior_data : pd.DataFrame
        Raw tracking DataFrame with a ``TIME (S)`` column.
    frequency : str
        Pandas resample frequency string, e.g. ``'1s'`` for 1-second bins.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame whose index is time in seconds (float).
    """
    list_of_column_names = list(behavior_data.columns)
    behavior_data_ds = pd.DataFrame()

    # Columns that encode discrete events are preserved with last() to avoid
    # averaging away brief activations within a bin.
    last_columns = {"IN PLATFORM", "IN REWARD ZONE", "IN CENTER",
                    "NEW SPEAKER ACTIVE", "SHOCKER ON ACTIVE"}

    for i in range(1, len(list_of_column_names)):
        col = list_of_column_names[i]
        series = behavior_data.set_index(
            pd.to_timedelta(behavior_data["TIME (S)"], unit="s")
        )[col]

        if col in last_columns:
            # Fill NaNs before resampling so brief events are not dropped.
            output = series.fillna(0).resample(frequency).last()
        else:
            output = series.resample(frequency).mean()

        output.bfill(inplace=True)
        output.index = output.index.total_seconds()
        behavior_data_ds[col] = output

    return behavior_data_ds


def process_data(df, phase):
    """Encode raw location and stimulus columns into state-ready features.

    Extracts the three location indicators (platform, center, reward zone)
    and constructs time-varying light, tone, and shock level columns based
    on the stimulus onset tables for the given ``phase``.

    Each stimulus is encoded as a graded level over its 30-second window:
    - **Light**: level 1 (s 0–9), 2 (s 10–19), 3 (s 20–29) after onset
    - **Tone**:  level 1 (s 0–14), 2 (s 15–24), 3 (s 25–29) after onset
    - **Shock**: level 1 for a 5-second window around onset (s −3 to +1)

    Parameters
    ----------
    df : pd.DataFrame
        Downsampled tracking DataFrame (output of ``downsample_behavior_data``).
    phase : str
        Experimental phase key: ``"RT"``, ``"LC"``, or ``"UNP"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``IN PLATFORM``, ``IN CENTER``, ``IN REWARD ZONE``,
        ``light_onset``, ``tone_onset``, ``shock_onset``.
    """
    df = df[["IN PLATFORM", "IN CENTER", "IN REWARD ZONE"]]

    # --- light level (graded over 30 s) ---
    light_onset = pd.DataFrame(
        {"light_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in LIGHT_ONSET[phase]:
        light_onset.loc[t     : t +  9, "light_onset"] = 1  # early
        light_onset.loc[t + 10: t + 19, "light_onset"] = 2  # mid
        light_onset.loc[t + 20: t + 29, "light_onset"] = 3  # late

    # --- tone level (graded over 30 s) ---
    tone_onset = pd.DataFrame(
        {"tone_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in TONE_ONSET[phase]:
        tone_onset.loc[t     : t + 14, "tone_onset"] = 1   # early
        tone_onset.loc[t + 15: t + 24, "tone_onset"] = 2   # mid
        tone_onset.loc[t + 25: t + 29, "tone_onset"] = 3   # late (shock imminent)

    # --- shock level (binary, 5-s window) ---
    shock_onset = pd.DataFrame(
        {"shock_onset": np.zeros(df.shape[0], dtype=np.int_)}, index=df.index
    )
    for t in SHOCK_ONSET[phase]:
        shock_onset.loc[t - 3: t + 1, "shock_onset"] = 1

    features = pd.concat([light_onset, tone_onset, shock_onset], axis=1)

    df2 = pd.concat(
        [df.reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    df2.set_index(df.index, inplace=True)

    return df2


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

class StateAxis(IntEnum):
    """Indices into the composite state vector ``[location, light, tone]``.

    The SARSA state is a length-3 integer array whose axes are:

    - ``Loc``   (axis 0): animal's current location (see ``Location``)
    - ``Light`` (axis 1): light level at current time step (0 = off, 1–3 = on)
    - ``Tone``  (axis 2): tone level at current time step (0 = off, 1–3 = on)
    """
    Loc   = 0
    Light = 1
    Tone  = 2


class Location(IntEnum):
    """Discrete location codes used in the ``Loc`` axis of the state vector.

    - ``P`` (0): Platform — safe zone; reaching it during tone avoids shock
    - ``C`` (1): Center   — neutral corridor
    - ``R`` (2): Reward zone — delivers liquid reward when light is on
    """
    P = 0  # platform (avoidance zone)
    C = 1  # center
    R = 2  # reward zone


def row_to_state(row):
    """Convert one row of a processed DataFrame to a composite state vector.

    Reads the binary location flags and pre-computed stimulus level columns
    to produce a length-3 integer array ``[location, light_level, tone_level]``
    compatible with the SARSA Q-table axes defined by ``rt_lc_unp_state_spec``.

    Parameters
    ----------
    row : pd.Series
        A single row from the output of ``process_data``, containing columns
        ``IN PLATFORM``, ``IN REWARD ZONE``, ``light_onset``, and ``tone_onset``.

    Returns
    -------
    np.ndarray
        Shape ``(3,)``, dtype ``int``.  Index with ``StateAxis`` members.
    """
    s = np.zeros(3, dtype=int)

    # Determine location: platform takes priority over reward zone.
    if row["IN PLATFORM"] > 0:
        s[StateAxis.Loc] = Location.P
    elif row["IN REWARD ZONE"] > 0:
        s[StateAxis.Loc] = Location.R
    else:
        s[StateAxis.Loc] = Location.C  # default: center

    # Copy pre-computed stimulus levels directly.
    s[StateAxis.Light] = row["light_onset"]
    s[StateAxis.Tone]  = row["tone_onset"]

    return s
