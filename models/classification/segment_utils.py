import numpy as np
from typing import List, Tuple, Optional
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader

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

def split_groups_train_val_test(groups: np.ndarray,
                                train_size=0.8, val_size=0.1, test_size=0.1,
                                random_state=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    uniq = np.array(sorted(set(groups.astype(str))))
    gss1 = GroupShuffleSplit(test_size=(1 - train_size), random_state=random_state)
    train_idx, tmp_idx = next(gss1.split(uniq, groups=uniq))
    train_groups = set(uniq[train_idx]); tmp_groups = set(uniq[tmp_idx])

    # split tmp into val/test by ratio
    tmp = np.array(sorted(tmp_groups))
    if len(tmp) == 0:
        return train_groups, set(), set()
    tv = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(test_size=(1 - tv), random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(tmp, groups=tmp))
    val_groups = set(tmp[val_idx]); test_groups = set(tmp[test_idx])
    return train_groups, val_groups, test_groups


def build_segment_loaders_from_splits(X: np.ndarray, y_enc: dict, groups: np.ndarray,
                                      task: str, batch_size: int,
                                      segment_by: str = 'behavior'):
    

    use_gesture_for_seg = (segment_by == 'behavior_gesture')
    segs_all = contiguous_segments(groups.astype(str),
                                   y_enc['behavior'],
                                   y_enc['gesture'] if use_gesture_for_seg else None)

    train_g, val_g, test_g = split_groups_train_val_test(groups)
    segs_tr  = [s for s in segs_all if s[2] in train_g]
    segs_val = [s for s in segs_all if s[2] in val_g]
    segs_te  = [s for s in segs_all if s[2] in test_g]

    ds_tr  = SegmentDataset(X, y_enc, groups, segs_tr,  task=task)
    ds_val = SegmentDataset(X, y_enc, groups, segs_val, task=task)
    ds_te  = SegmentDataset(X, y_enc, groups, segs_te,  task=task)

    train_loader = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: pad_collate(b, task=task))
    val_loader   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: pad_collate(b, task=task))
    test_loader  = DataLoader(ds_te,  batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: pad_collate(b, task=task))

    return train_loader, val_loader, test_loader, (train_g, val_g, test_g)