"""
tint.visualization
==================

Visualization tools for tracks objects.

"""

import gc

from matplotlib import animation
from matplotlib import pyplot as plt
import pyart

from .grid_utils import get_grid_alt


def animate(tobj, grids, outfile_name, isolated_only=False, fps=1):
    """
    Creates gif animation of tracked cells.

    Parameters
    ----------
    tobj : Cell_tracks
        The Cell_tracks object to be visualized.
    grids : iterable
        An iterable containing all of the grids used to generate tobj
    outfile_name : str
        The name of the output file to be produced.
    arrows : bool
        If True, draws arrow showing corrected shift for each object.
    isolation : bool
        If True, only annotates uids for isolated objects.
    fps : int
        Frames per second for output gif.

    """

    grid_size = tobj.grid_size
    nframes = tobj.tracks.index.levels[0].max() + 1
    print('Animating', nframes, 'frames')

    def animate_frame(enum_grid):
        """ Animate a single frame of gridded reflectivity including
        uids. """
        plt.clf()
        nframe, grid = enum_grid
        print('Frame:', nframe)
        display = pyart.graph.GridMapDisplay(grid)
        ax = fig_grid.add_subplot(111)
        display.plot_basemap()
        gs_alt = tobj.params['GS_ALT']
        display.plot_grid(tobj.field, level=get_grid_alt(grid_size, gs_alt),
                          vmin=-8, vmax=64, mask_outside=False,
                          cmap=pyart.graph.cm.NWSRef)

        if nframe in tobj.tracks.index.levels[0]:
            frame_tracks = tobj.tracks.loc[nframe]
            for ind, uid in enumerate(frame_tracks.index):

                if isolated_only and not frame_tracks['isolated'].iloc[ind]:
                    continue
                x = frame_tracks['grid_x'].iloc[ind]*grid_size[2]
                y = frame_tracks['grid_y'].iloc[ind]*grid_size[1]
                ax.annotate(uid, (x, y), fontsize=20)

        del grid, enum_grid, display, ax, frame_tracks
        gc.collect()

    fig_grid = plt.figure(figsize=(10, 8))
    anim_grid = animation.FuncAnimation(fig_grid, animate_frame,
                                        frames=enumerate(grids))
    anim_grid.save(outfile_name,
                   writer='imagemagick', fps=fps)
    plt.close()
