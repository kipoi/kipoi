import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from concise.utils.plot import add_letter_to_axis, VOCABS, letter_polygons
from mpl_toolkits.axes_grid1 import make_axes_locatable



def center_cmap(cmap, vmax, vmin, center):
    ### Centering of the colormap, taken from seaborn._HeatMapper implementation
    import matplotlib as mpl
    vrange = max(vmax - center, center - vmin)
    normlize = mpl.colors.Normalize(center - vrange, center + vrange)
    cmin, cmax = normlize([vmin, vmax])
    cc = np.linspace(cmin, cmax, 256)
    return  mpl.colors.ListedColormap(cmap(cc))





def seqlogo_heatmap(letter_heights, heatmap_data, vocab="DNA", ax=None, show_letter_scale = False, cmap = plt.cm.bwr,
                    cbar = True, cbar_kws = None, cbar_ax = None):
    """
    Plot heatmap and seqlogo plot together in one axis.
    
    # Arguments
        letter_heights: "sequence length" x "vocabulary size" numpy array of seqlogo letter heights
        heatmap_data: "vocabulary size" x "sequence length" numpy array of heatmap values
    Can also contain negative values.
        vocab: str, Vocabulary name. Can be: DNA, RNA, AA, RNAStruct.
        ax: matplotlib axis
    """
    assert heatmap_data.shape[1] == letter_heights.shape[0]
    seq_len = heatmap_data.shape[1]
    vocab_len = len(VOCABS[vocab])
    letter_rescaling = len(VOCABS[vocab]) # This way the heatmap and the letters are the same size on the plot

    # heatmap grid
    grid = np.mgrid[0.5:(seq_len + 0.5):1, -vocab_len:0:1].reshape(2, -1).T
    y_hm_tickpos = np.arange(-vocab_len, 0, 1) + 0.5
    y_seqlogo_tickpos = np.array([0, letter_rescaling])  # tuple of where the ticks for the seqlogo should be placed

    if ax is None:
        plt.figure(figsize = (20,4))
        ax = plt.subplot(1,1,1)
    patches = []
    # add a circle
    for pos_tuple in grid:
        rect = mpatches.Rectangle(pos_tuple, 1.0,1.0,ec="none")
        patches.append(rect)

    # Add colours to the heatmap
    colors = heatmap_data.T.reshape((seq_len *4))
    # Centre the colours around 0
    cmap_centered = center_cmap(cmap, colors.max(), colors.min(), 0.0)
    collection = PatchCollection(patches, cmap=cmap_centered, alpha=1.0)
    collection.set_array(np.array(colors))

    # add the heatmap to the axis
    hm_ax_collection = ax.add_collection(collection)

    # rescale letters so that they look nice above the heatmap
    letter_heights_rescaled = np.copy(letter_heights)
    letter_heights_rescaled /= letter_heights_rescaled.max()/4.0

    assert letter_heights.shape[1] == len(VOCABS[vocab])
    x_range = [1, letter_heights.shape[0]]

    for x_pos, heights in enumerate(letter_heights_rescaled):
        letters_and_heights = sorted(zip(heights, list(VOCABS[vocab].keys())))
        y_pos_pos = 0.0
        y_neg_pos = 0.0
        for height, letter in letters_and_heights:
            color = VOCABS[vocab][letter]
            polygons = letter_polygons[letter]
            if height > 0:
                add_letter_to_axis(ax, polygons, color, 0.5 + x_pos, y_pos_pos, height)
                y_pos_pos += height
            else:
                add_letter_to_axis(ax, polygons, color, 0.5 + x_pos, y_neg_pos, height)
                y_neg_pos += height

    ax.set_xlim(x_range[0] - 1, x_range[1] + 1)
    ax.grid(False)
    ax.set_xticks(list(range(*x_range)) + [x_range[-1]])
    ax.set_aspect(aspect='auto', adjustable='box')

    # set the tick labels and make sure only the left axis is displayed
    if show_letter_scale:
        y_ticks = np.concatenate([y_hm_tickpos, y_seqlogo_tickpos])
        yticklabels = list(VOCABS[vocab].keys()) + ["%.2f"%letter_heights.min(), "%.2f"%letter_heights.max()]
        ax.spines['left'].set_bounds(y_seqlogo_tickpos[0], y_seqlogo_tickpos[1])
    else:
        y_ticks = y_hm_tickpos
        yticklabels = list(VOCABS[vocab].keys())
        ax.spines['left'].set_visible(False)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(yticklabels)
    ax.axes.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.autoscale_view()
    if cbar:
        ### Colorbar settings adapted from seaborn._HeatMapper implementation
        import matplotlib as mpl
        cbar_kws = {} if cbar_kws is None else cbar_kws
        cbar_kws.setdefault('ticks', mpl.ticker.MaxNLocator(6))
        cb = ax.figure.colorbar(hm_ax_collection, cbar_ax, ax, **cbar_kws)
        cb.outline.set_linewidth(0)
        # If rasterized is passed to pcolormesh, also rasterize the
        # colorbar to avoid white lines on the PDF rendering
        #if kws.get('rasterized', False):
        cb.solids.set_rasterized(True)
    return ax

