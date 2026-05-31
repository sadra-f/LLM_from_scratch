import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from typing import List, Optional, Tuple, Union
import math
from typing import List, Optional
import torch

#Most of the code here is AI written sorry <3

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    # if ax is None:
    ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_attention_heatmaps(
    attention_weights: List[np.ndarray],
    row_col_values: List[str],
    title: Optional[str] = None,
    layer_names: Optional[List[str]] = None,
    max_tokens: int = 32,
    cmap: str = "magma",
):
    """
    Visualize transformer attention.

    Expected input per layer:

        attention_weights[layer]
            shape = (num_heads, seq_len, seq_len)

    Example:
        layer_att = weights[layer][0].detach().cpu().numpy()

    where:
        weights[layer] shape:
            (batch_size, num_heads, seq_len, seq_len)

    and:
        attention_weights.append(weights[layer][0])
    """

    if len(attention_weights) == 0:
        raise ValueError("attention_weights cannot be empty")

    if layer_names is None:
        layer_names = [
            f"Layer {i + 1}"
            for i in range(len(attention_weights))
        ]

    tokens = row_col_values[-max_tokens:]

    for layer_idx, layer_att in enumerate(attention_weights):

        if isinstance(layer_att, torch.Tensor):
            layer_att = layer_att.detach().cpu().numpy()

        if layer_att.ndim != 3:
            raise ValueError(
                f"Layer {layer_idx} expected shape "
                f"(heads, seq, seq) but got {layer_att.shape}"
            )

        layer_att = layer_att[
            :,
            -max_tokens:,
            -max_tokens:
        ]

        num_heads = layer_att.shape[0]

        ncols = min(4, num_heads)
        nrows = math.ceil(num_heads / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4 * ncols, 4 * nrows)
        )

        axes = np.array(axes).reshape(-1)

        vmin = layer_att.min()
        vmax = layer_att.max()

        for head_idx in range(num_heads):

            ax = axes[head_idx]

            im = ax.imshow(
                layer_att[head_idx],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                interpolation="nearest"
            )

            ax.set_title(
                f"Head {head_idx}",
                fontsize=10
            )

            tick_step = max(
                1,
                len(tokens) // 8
            )

            ticks = list(
                range(
                    0,
                    len(tokens),
                    tick_step
                )
            )

            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            ax.set_xticklabels(
                [tokens[i] for i in ticks],
                rotation=90,
                fontsize=7
            )

            ax.set_yticklabels(
                [tokens[i] for i in ticks],
                fontsize=7
            )

        for idx in range(num_heads, len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            layer_names[layer_idx],
            fontsize=16,
            fontweight="bold"
        )

        fig.subplots_adjust(
            right=0.90,
            hspace=0.30,
            wspace=0.25
        )

        cbar_ax = fig.add_axes(
            [0.92, 0.15, 0.02, 0.70]
        )

        fig.colorbar(
            im,
            cax=cbar_ax
        )

        plt.show()