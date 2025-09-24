import numpy as np
from typing import List, Tuple, Optional
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

# ---- variable-length segment dataset (per contiguous action) ----
class SegmentDataset(Dataset):
    """
    Yields variable-length segments as (C, T), with labels.
    segments: list of (start, end, group_key, behavior_id, gesture_id)
    """
    def __init__(self, X: np.ndarray, y_enc: dict, groups: np.ndarray, segments: list, task: str='both'):
        self.X = X
        self.y = y_enc
        self.groups = groups
        self.segments = segments
        self.task = task

    def __len__(self): return len(self.segments)

    def __getitem__(self, i):
        s, e, gk, b, ge = self.segments[i]
        x = self.X[s:e].astype(np.float32).T  # (C, T)
        if self.task == 'behavior':
            y = torch.tensor(b, dtype=torch.long)
        elif self.task == 'gesture':
            y = torch.tensor(ge, dtype=torch.long)
        else:
            y = (torch.tensor(b, dtype=torch.long), torch.tensor(ge, dtype=torch.long))
        return torch.from_numpy(x), y

def pad_collate(batch, task='both'):
    """
    Pad variable T to T_max so Conv1d can run.
    Returns (x_pad: B,C,T_max, y, lengths)
    """
    xs, ys = zip(*batch)
    C = xs[0].shape[0]
    lengths = [x.shape[1] for x in xs]
    Tm = max(lengths)
    x_pad = torch.zeros(len(xs), C, Tm, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        t = x.shape[1]
        x_pad[i, :, :t] = x
    if task == 'both':
        yb = torch.stack([y[0] for y in ys], dim=0)
        yg = torch.stack([y[1] for y in ys], dim=0)
        y = (yb, yg)
    else:
        y = torch.stack(list(ys), dim=0)
    return x_pad, y, torch.tensor(lengths, dtype=torch.int32)

def contiguous_segments(groups: np.ndarray,
                        behavior: np.ndarray,
                        gesture: Optional[np.ndarray] = None) -> List[Tuple[int,int,str,int,int]]:
    """
    Returns a list of segments (start, end, group_key, behavior_id, gesture_id)
    where group stays constant and labels don't change. 'end' is exclusive.
    """

    N = len(groups)
    out = []
    s = 0

    for i in range(1, N + 1):
        boundary = (i == N) \
        or (groups[i] != groups[i-1]) \
                   or (behavior[i] != behavior[i-1]) \
                   or (gesture is not None and gesture[i] != gesture[i-1])
        if boundary:
            gk = groups[s]
            b  = behavior[s]
            ge = gesture[s] if gesture is not None else -1
            out.append((s, i, gk, int(b), int(ge)))
            s = i
    return out