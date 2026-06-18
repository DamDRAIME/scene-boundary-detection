from enum import StrEnum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from labellines import labelLines

from sbd.dtw.dtw import DTWOutput
from sbd.dtw.utils import gaussian_blur


class PlotType(StrEnum):
    ALIGNMENT = auto()
    CONTOUR = auto()
    TWO_WAY = auto()
    THREE_WAY = auto()


def plot(
    dtw: DTWOutput,
    dst_filepath: Path,
    type: PlotType = PlotType.CONTOUR,
    x_label: str = "Query",
    y_label: str = "Reference",
    display_optimal_path: bool = True,
) -> Path:
    assert (
        dtw.cost_matrix
    ), "Plotting the results of the DTW, requires DTW's artifacts. Set `keep_artifacts` to True when using `dtw()`"

    cm = dtw.cost_matrix
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(6, 6))

    # Heatmap
    ax.imshow(cm.T, origin="lower", cmap=sns.cubehelix_palette(as_cmap=True))

    # Contour - The Cost Matrix is blurred to have smoother contour lines
    cm_blurred = gaussian_blur(cm.T)
    co = ax.contour(cm_blurred, colors="#5c5c5c", linewidths=1, linestyles="solid")
    ax.clabel(co)

    # Optimal path
    if display_optimal_path:
        ax.plot(
            *zip(*dtw.optimal_warping_path), color="#392b3b", linewidth=2, linestyle="dashed", label="optimal path"
        )
        labelLines(plt.gca().get_lines(), align=True)  # Align label along line

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.savefig(dst_filepath)
    return dst_filepath


# import numpy as np

# _idx = np.linspace(0, 6.28, num=300)
# x = (np.sin(_idx) + np.random.uniform(0, 0.1, len(_idx)),)
# y = np.cos(_idx)
# x = np.stack([x, np.ones_like(x)], axis=-1).squeeze()
# y = np.stack([y, np.ones_like(y)], axis=-1).squeeze()
# a = torch.Tensor(x)
# b = torch.Tensor(y)
# res = dtw(a, b, DistanceFunction.EUCLIDEAN, keep_artifacts=True)
# # print(res.optimal_warping_path)
# plot(res, "conout.png")
