"""
Raw EEG Data Loader  -  BrainFlow Cyton WiFi format
====================================================

Reads  data/alpha_wave_arm_data/alpha-training-task/s{N}_d{M}_training.txt
and produces labeled windows ready for the CNN classification pipeline.

File format  (23 comma-separated columns, NO header):
  col 0     : sample_index
  col 1-8   : EEG channels  [POz, PO3, PO4, PO7, PO8, O1, O2, Oz]  (raw ADC)
  col 9-11  : Accelerometer [accel_x, accel_y, accel_z]
  col 12-20 : Board status / aux bytes
  col 21    : Unix timestamp

Labeling modes
--------------
'block'
    Each recording is divided into N_TASKS equal-length blocks and assigned the
    task labels in TASK_ORDER. No external timing file needed. Assumes the
    experimenter ran push → pull → pick → stack in that order with roughly
    equal time per task.

'excel'
    Reads a timing Excel file (trial-completion_selection-performance.xlsx or
    similar) that contains columns  [subject, day, task, start_time, end_time].
    Windows whose centre timestamp falls in a task interval are labeled
    accordingly.

Output
------
X      : np.ndarray  (N_windows, 8, T_window)  float32  – CNN input (B, C, T)
y      : dict  {'behavior': np.ndarray (N_windows,) int}
groups : np.ndarray  (N_windows,) str  – e.g. 's1_d1'  for GroupShuffleSplit
meta   : pd.DataFrame  with per-window metadata (subject, day, start_sample, label_str)
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── Constants ────────────────────────────────────────────────────────────────

EEG_COLS      = ['POz', 'PO3', 'PO4', 'PO7', 'PO8', 'O1', 'O2', 'Oz']
ACCEL_COLS    = ['accel_x', 'accel_y', 'accel_z']
N_OTHER       = 9
OTHER_COLS    = [f'other_{i}' for i in range(N_OTHER)]
TIMESTAMP_COL = 'timestamp'
ALL_COLS      = (['sample_index'] + EEG_COLS + ACCEL_COLS
                 + OTHER_COLS + [TIMESTAMP_COL])

CYTON_SCALE = (4.5 / (2**23 - 1)) / 24 * 1e6  

FS = 250.0   # Hz

TASK_ORDER: List[str] = ['push', 'pull', 'pick', 'stack']


# ── Low-level loader ─────────────────────────────────────────────────────────

def _parse_subject_day(filename: str) -> Tuple[str, str]:
    m = re.match(r'(s\d+)_?(d\d+)', Path(filename).stem)
    return (m.group(1), m.group(2)) if m else ('unknown', 'unknown')


def load_txt(path: str | Path, scale_to_uv: bool = True) -> pd.DataFrame:
    """
    Load one BrainFlow Cyton WiFi .txt file.

    Returns a DataFrame with named columns. EEG values are optionally
    converted to μV with the Cyton hardware scale factor.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=',', header=None, dtype=float,
                     on_bad_lines='skip')
    n = df.shape[1]
    names = ALL_COLS[:n]
    if len(names) < n:
        names += [f'extra_{i}' for i in range(n - len(names))]
    df.columns = names

    if scale_to_uv:
        eeg_present = [c for c in EEG_COLS if c in df.columns]
        df[eeg_present] = df[eeg_present] * CYTON_SCALE

    # Remove NaN rows
    df = df.dropna(subset=[c for c in EEG_COLS if c in df.columns])
    df = df.reset_index(drop=True)
    return df


def load_all_txt(data_dir: str | Path,
                 subject_ids: Optional[List[str]] = None,
                 scale_to_uv: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load all .txt files in data_dir.

    Args
    ----
    data_dir   : directory containing s{N}_d{M}_training.txt files
    subject_ids: e.g. ['s1', 's2']  (None = all)

    Returns
    -------
    dict mapping 's1_d1' → DataFrame
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob('*.txt'))
    dfs: Dict[str, pd.DataFrame] = {}

    for f in files:
        subj, day = _parse_subject_day(f.name)
        if subject_ids and subj not in subject_ids:
            continue
        key = f'{subj}_{day}'
        try:
            dfs[key] = load_txt(f, scale_to_uv=scale_to_uv)
            print(f'  Loaded {f.name}: {len(dfs[key]):,} samples')
        except Exception as e:
            warnings.warn(f'Could not load {f.name}: {e}')
    return dfs


# ── Windowing ────────────────────────────────────────────────────────────────

def make_windows(eeg: np.ndarray, window: int, stride: int) -> np.ndarray:
    """
    eeg    : (N_samples, 8)  EEG matrix
    window : samples per window
    stride : stride in samples

    Returns (N_windows, 8, window)  - CNN-ready (B, C, T)
    """
    starts = range(0, len(eeg) - window + 1, stride)
    if not starts:
        return np.empty((0, eeg.shape[1], window), dtype=np.float32)
    segs = np.stack([eeg[s:s + window].T for s in starts]).astype(np.float32)
    return segs   # (W, 8, window)


# ── Block labeler ────────────────────────────────────────────────────────────

def label_by_block(df: pd.DataFrame,
                   task_order: List[str] = TASK_ORDER) -> pd.Series:
    """
    Divide the recording into len(task_order) equal-duration blocks and assign
    the task name as label. Returns a Series of str labels, same length as df.
    """
    n = len(df)
    block_size = n // len(task_order)
    labels = pd.Series([''] * n, dtype=str)
    for i, task in enumerate(task_order):
        start = i * block_size
        end = (i + 1) * block_size if i < len(task_order) - 1 else n
        labels.iloc[start:end] = task
    return labels


# ── Excel timing labeler ─────────────────────────────────────────────────────

def label_by_excel(df: pd.DataFrame, excel_path: str | Path,
                   subject: str, day: str) -> pd.Series:
    """
    Assign task labels using trial timing from an Excel file.

    The Excel file must contain columns:
        subject  : str  (e.g. 's1')
        day      : str  (e.g. 'd1')
        task     : str  (e.g. 'push')
        start    : float  Unix timestamp
        end      : float  Unix timestamp

    Rows whose timestamp is outside all labelled intervals get label 'rest'.
    """
    timing = pd.read_excel(excel_path)
    # Normalise column names
    timing.columns = timing.columns.str.strip().str.lower()
    required = {'subject', 'day', 'task', 'start', 'end'}
    missing = required - set(timing.columns)
    if missing:
        raise ValueError(
            f'Excel file missing columns: {missing}. '
            f'Found: {list(timing.columns)}')

    sub_timing = timing[
        (timing['subject'].astype(str) == subject) &
        (timing['day'].astype(str) == day)
    ]

    labels = pd.Series(['rest'] * len(df), dtype=str)
    ts = df[TIMESTAMP_COL].values if TIMESTAMP_COL in df.columns else None
    if ts is None:
        warnings.warn('No timestamp column found - falling back to block labelling.')
        return label_by_block(df)

    for _, row in sub_timing.iterrows():
        mask = (ts >= row['start']) & (ts <= row['end'])
        labels[mask] = str(row['task']).strip().lower()

    return labels


# ── Main API ─────────────────────────────────────────────────────────────────

def load_eeg_dataset(
    data_dir:       str | Path,
    window_sec:     float = 1.0,
    overlap:        float = 0.5,
    subject_ids:    Optional[List[str]] = None,
    labeling:       str = 'block',
    task_order:     List[str] = TASK_ORDER,
    excel_path:     Optional[str | Path] = None,
    scale_to_uv:    bool = True,
    exclude_rest:   bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """
    Load and window all EEG recordings, returning training-ready arrays.

    Args
    ----
    data_dir     : folder containing .txt files
    window_sec   : window duration in seconds (default 1.0 → 250 samples)
    overlap      : fraction of window overlap (0-1, default 0.5)
    subject_ids  : list of subject IDs to include (None = all)
    labeling     : 'block' or 'excel'
    task_order   : task label order for 'block' mode
    excel_path   : path to timing Excel file for 'excel' mode
    scale_to_uv  : convert raw ADC → μV (recommended)
    exclude_rest : skip windows labelled 'rest' (default True)

    Returns
    -------
    X      : (N, 8, T)  float32  EEG windows
    y      : {'behavior': (N,) int}  label-encoded action labels
    groups : (N,) str  subject+day identifier for GroupShuffleSplit
    meta   : DataFrame with columns [subject, day, window_start, label_str]
    """
    data_dir = Path(data_dir)
    window   = int(window_sec * FS)
    stride   = int(window * (1.0 - overlap))

    dfs = load_all_txt(data_dir, subject_ids=subject_ids,
                       scale_to_uv=scale_to_uv)
    if not dfs:
        raise FileNotFoundError(f'No .txt files loaded from {data_dir}')

    all_X:     List[np.ndarray] = []
    all_lab:   List[str]        = []
    all_group: List[str]        = []
    all_meta:  List[dict]       = []

    for key, df in dfs.items():
        subj, day = key.split('_')
        eeg = df[EEG_COLS].values  # (N, 8)

        # Get per-sample labels
        if labeling == 'excel':
            if excel_path is None:
                raise ValueError(
                    "excel_path must be provided when labeling='excel'")
            sample_labels = label_by_excel(df, excel_path, subj, day)
        else:
            sample_labels = label_by_block(df, task_order)

        # Window
        starts = range(0, len(eeg) - window + 1, stride)
        for s in starts:
            win_eeg = eeg[s:s + window].T.astype(np.float32)  # (8, T)
            # Label = majority vote over the window
            win_labels = sample_labels.iloc[s:s + window]
            label = win_labels.value_counts().idxmax()

            if exclude_rest and label == 'rest':
                continue

            all_X.append(win_eeg)
            all_lab.append(label)
            all_group.append(key)
            all_meta.append({
                'subject': subj, 'day': day,
                'window_start': s, 'label_str': label
            })

    if not all_X:
        raise ValueError('No valid windows found. Check data_dir and labeling mode.')

    X = np.stack(all_X)                     # (N, 8, T)
    label_strs = np.array(all_lab)
    groups     = np.array(all_group)
    meta       = pd.DataFrame(all_meta)

    le = LabelEncoder()
    y_int = le.fit_transform(label_strs)
    y = {'behavior': y_int, 'gesture': np.zeros(len(y_int), dtype=np.int64)}

    print(f'\nLoaded EEG dataset:')
    print(f'  Windows : {len(X):,}')
    print(f'  Shape   : {X.shape}  (N, C, T)')
    print(f'  Subjects: {sorted(np.unique(meta["subject"].values))}')
    print(f'  Labels  : {dict(zip(le.classes_, [int((y_int==i).sum()) for i in range(len(le.classes_))]))  }')

    # Store encoder on y dict for convenience
    y['encoder']   = le
    y['label_str'] = label_strs

    return X, y, groups, meta


# ── Convenience wrapper matching data_utilities API ──────────────────────────

def load_sensor_data_raw(
    data_dir:     str | Path,
    window_sec:   float = 1.0,
    overlap:      float = 0.5,
    subject_ids:  Optional[List[str]] = None,
    labeling:     str = 'block',
    task_order:   List[str] = TASK_ORDER,
    excel_path:   Optional[str | Path] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """
    Drop-in replacement for data_utilities.load_sensor_data() that reads
    raw .txt files instead of a pre-processed CSV.

    Returns
    -------
    X_raw  : (N_windows, 8*T_window)  flat float32  (matches data_utilities shape)
    y      : {'behavior': str array, 'gesture': str array}
    groups : str array
    meta   : DataFrame

    Note: The classification pipeline's preprocess_multilabel() will
          StandardScale X_raw. For raw EEG, the input to the CNN pipeline
          is X_raw.reshape(N, 8, T_window) not the flat version.
          Use load_eeg_dataset() directly when possible.
    """
    X, y_int, groups, meta = load_eeg_dataset(
        data_dir=data_dir,
        window_sec=window_sec,
        overlap=overlap,
        subject_ids=subject_ids,
        labeling=labeling,
        task_order=task_order,
        excel_path=excel_path,
        scale_to_uv=True,
        exclude_rest=True,
    )
    le = y_int.pop('encoder')
    y_int.pop('label_str')

    # Convert int labels back to str (data_utilities convention)
    y_str = {
        'behavior': le.inverse_transform(y_int['behavior']),
        'gesture':  np.array(['none'] * len(y_int['behavior']), dtype=object),
    }
    # Flatten X for compatibility with data_utilities (pipeline will reshape)
    X_flat = X.reshape(len(X), -1).astype(np.float32)   # (N, 8*T)
    return X_flat, y_str, groups, meta


# ── ROS bag helper info ──────────────────────────────────────────────────────

ROS_BAG_TOPICS_HINT = """
ROS Bag Usage Guide
===================
The bags at data/alpha_wave_arm_data/ROS-message-data/s{N}_d{M}/ contain
synchronized robot+EEG data from the experiments.

To extract data from a ROS1 bag (*.bag):
    pip install rosbag_pandas    # or  pip install bagpy

    import bagpy
    b = bagpy.bagreader('system_dump_2020-06-04-14-58-42.bag')
    # List topics
    print(b.topic_table)
    # Read a topic as DataFrame
    df = b.message_by_topic('/joint_states')

Common topics to look for:
  /joint_states       → robot joint positions, velocities  (for RL state)
  /cmd_vel / /cmd     → robot commands / actions           (for imitation learning)
  /camera/image_raw   → camera images                      (for VLA training)
  /eeg                → raw EEG if streamed over ROS       (for synchronisation)
  /task_label         → task identifier                    (for labelling EEG)

For VLA fine-tuning (OpenVLA / Octo-style):
  - Extract (image_t, instruction_str, action_t) triplets
  - Instruction: TASK_TO_TEXT[task_label] from sequential_vla_agent.py
  - Action: delta joint position from consecutive /joint_states messages

For RL training (REPPO):
  - Extract (obs_t, action_t, obs_{t+1}) and compute dense rewards
  - Use openarm_mj_env.py reward functions as a reference

For EEG classification labelling:
  - Read /task_label topic timestamps and match against EEG timestamps
  - This is more accurate than the block-based heuristic above
"""

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/alpha_wave_arm_data/alpha-training-task')
    p.add_argument('--window-sec', type=float, default=1.0)
    p.add_argument('--overlap', type=float, default=0.5)
    p.add_argument('--labeling', choices=['block', 'excel'], default='block')
    p.add_argument('--excel-path', default=None)
    p.add_argument('--subjects', nargs='+', default=None)
    args = p.parse_args()

    X, y, groups, meta = load_eeg_dataset(
        data_dir=args.data_dir,
        window_sec=args.window_sec,
        overlap=args.overlap,
        subject_ids=args.subjects,
        labeling=args.labeling,
        excel_path=args.excel_path,
    )
    print(f'\nX shape: {X.shape}')
    print(f'Labels: {np.unique(y["label_str"])}')
    print(ROS_BAG_TOPICS_HINT)
