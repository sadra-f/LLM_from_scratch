import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

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


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union

def plot_attention_heatmaps(
    attention_weights: List[np.ndarray],
    row_col_values: List[str],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    separate_windows: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    layer_names: Optional[List[str]] = None,
    annot: bool = False,
    fmt: str = '.2f',
    cbar: bool = True
) -> Union[plt.Figure, List[plt.Figure]]:
    """
    Plot attention weights from different layers as heatmaps.
    
    Parameters:
    -----------
    attention_weights : List[np.ndarray]
        List of attention weight matrices. Each element should be an M x M numpy array
        where M is the number of tokens/positions.
    
    row_col_values : List[str]
        List of strings of length M representing the labels for rows and columns.
    
    title : str, optional
        Overall title for the figure (only applies when separate_windows=False).
    
    figsize : Tuple[int, int], default=(12, 10)
        Size of each figure/window.
    
    cmap : str, default='viridis'
        Colormap for the heatmap.
    
    separate_windows : bool, default=False
        If True, each layer gets its own figure window.
        If False, all layers are plotted as subplots in a single figure.
    
    vmin, vmax : float, optional
        Minimum and maximum values for colormap normalization.
        If None, use the min/max of the data.
    
    layer_names : List[str], optional
        List of names for each layer. If None, uses 'Layer i'.
    
    annot : bool, default=False
        If True, write the data value in each cell.
    
    fmt : str, default='.2f'
        Format string for annotations when annot=True.
    
    cbar : bool, default=True
        Whether to show colorbar.
    
    Returns:
    --------
    Union[plt.Figure, List[plt.Figure]]
        If separate_windows=False, returns the single Figure object.
        If separate_windows=True, returns a list of Figure objects.
    """
    
    n_layers = len(attention_weights)
    
    if not attention_weights:
        raise ValueError("attention_weights list cannot be empty")
    
    # Check dimensions
    M = len(row_col_values)
    for i, weights in enumerate(attention_weights):
        if weights.shape != (M, M):
            raise ValueError(f"Layer {i} weights shape {weights.shape} doesn't match M={M}")
    
    # Set default layer names
    if layer_names is None:
        layer_names = [f'Layer {i+1}' for i in range(n_layers)]
    else:
        if len(layer_names) != n_layers:
            raise ValueError("layer_names length must match attention_weights length")
    
    # Set vmin/vmax if not provided
    if vmin is None:
        vmin = min([w.min() for w in attention_weights])
    if vmax is None:
        vmax = max([w.max() for w in attention_weights])
    
    if separate_windows:
        # Separate windows for each layer
        figures = []
        for i, weights in enumerate(attention_weights):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            
            # Create heatmap
            im = ax.imshow(weights, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Set labels
            ax.set_xticks(range(M))
            ax.set_yticks(range(M))
            ax.set_xticklabels(row_col_values, rotation=45, ha='right')
            ax.set_yticklabels(row_col_values)
            
            # Set title
            ax.set_title(layer_names[i], fontsize=14, fontweight='bold')
            
            # Add annotations if requested
            if annot:
                for i_idx in range(M):
                    for j_idx in range(M):
                        text = ax.text(j_idx, i_idx, format(weights[i_idx, j_idx], fmt),
                                     ha="center", va="center", color="w" if weights[i_idx, j_idx] > (vmax+vmin)/2 else "k")
            
            # Add colorbar
            if cbar:
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            figures.append(fig)
        
        # Show all figures
        plt.show()
        return figures
    
    else:
        # Single figure with subplots
        # Calculate grid dimensions
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows))
        
        # Handle case of single subplot
        if n_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each layer
        for i, weights in enumerate(attention_weights):
            ax = axes[i]
            
            # Create heatmap
            im = ax.imshow(weights, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Set labels
            ax.set_xticks(range(M))
            ax.set_yticks(range(M))
            ax.set_xticklabels(row_col_values, rotation=45, ha='right')
            ax.set_yticklabels(row_col_values)
            
            # Set title
            ax.set_title(layer_names[i], fontsize=12, fontweight='bold')
            
            # Add annotations if requested
            if annot:
                for i_idx in range(M):
                    for j_idx in range(M):
                        text = ax.text(j_idx, i_idx, format(weights[i_idx, j_idx], fmt),
                                     ha="center", va="center", color="w" if weights[i_idx, j_idx] > (vmax+vmin)/2 else "k",
                                     fontsize=8)
        
        # Hide unused subplots
        for i in range(n_layers, len(axes)):
            axes[i].axis('off')
        
        # Add overall title
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        # Add colorbar
        if cbar:
            fig.colorbar(im, ax=axes.tolist(), orientation='vertical', fraction=0.02)
        
        plt.tight_layout()
        plt.show()
        
        return fig
