import numpy as np
from typing import List, Tuple, Optional

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