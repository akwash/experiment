import numpy as np

# estimate rover psitiion for determining local coord system

def heading_from_path(path, current_step):
    if current_step >= len(path) - 1:
        # at goal, keep last heading instead of returning None
        curr = np.array(path[current_step - 1])
        next_ = np.array(path[current_step])
    else:
        curr = np.array(path[current_step]) # current step in path
        next_ = np.array(path[current_step + 1]) # next step in path

    delta = next_ - curr # position change (row, col)

    if np.all(delta == 0):
        return 0.0  # no movement, default heading

    heading = np.arctan2(delta[0], delta[1]) # angle from x axis, y is forward, x is left/right
    return heading