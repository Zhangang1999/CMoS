
import numpy as np

def flow_to_rgb(flow_map, max_value):
    _, h, w  = flow_map.shape

    flow_map[: (flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')

    rgb_map = np.ones((3, h, w)).astype(np.float32)

    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map.max()))
    
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]

    return rgb_map.clip(0, 1)