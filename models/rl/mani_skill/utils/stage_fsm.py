from enum import IntEnum

class Stage(IntEnum):
    push = 0
    pull = 1
    pick = 2
    stack = 3
    done = 4

DEFAULT_SEQ = [Stage.push, Stage.pull, Stage.pick, Stage.stack, Stage.done]