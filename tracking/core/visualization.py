from matplotlib import pyplot as plt
from matplotlib import animation
import pyart
import gc

from .grid_utils import get_grid_alt


def animate(tobj, grids, outfile_name, arrows=False, isolation=False, fps=1):
    """Creates gif animation of tracked cells."""
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

                if isolation and not frame_tracks['isolated'].iloc[ind]:
                    continue
                x = frame_tracks['grid_x'].iloc[ind]*grid_size[2]
                y = frame_tracks['grid_y'].iloc[ind]*grid_size[1]
                ax.annotate(uid, (x, y), fontsize=20)
                if arrows and ((nframe, uid) in tobj.record.shifts.index):
                    shift = tobj.record.shifts \
                        .loc[nframe, uid]['corrected']
                    shift = shift * grid_size[1:]
                    ax.arrow(x, y, shift[1], shift[0],
                             head_width=3*grid_size[1],
                             head_length=6*grid_size[1])
        del grid, enum_grid, display, ax, frame_tracks
        gc.collect()

    fig_grid = plt.figure(figsize=(10, 8))
    anim_grid = animation.FuncAnimation(fig_grid, animate_frame,
                                        frames=enumerate(grids))
    anim_grid.save(outfile_name,
                   writer='imagemagick', fps=fps)
    plt.close()
