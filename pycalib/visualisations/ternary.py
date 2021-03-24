import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
from .barycentric import bc2xy, xy2bc


def draw_tri_samples(pvals, classes, labels=None, fig=None, ax=None,
                     legend=True, color_list=[None]*3,
                     **kwargs):
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    if labels is None:
        labels = [r'$C_{}$'.format(i+1) for i in range(len(corners))]
    center = corners.mean(axis=0)
    for i, corner in enumerate(corners):
        text_x, text_y = corner - (center - corner)*0.1
        ax.text(text_x, text_y, labels[i], verticalalignment='center',
                horizontalalignment='center')

    xy = bc2xy(pvals, corners)

    # TODO Find option to call scatter only once as now the latter classes are
    # on top of the previous ones
    for c in [0, 1, 2]:
        c_idx = classes == c
        ax.scatter(xy[c_idx, 0], xy[c_idx, 1],
                   label=labels[c], color=color_list[c],
                   **kwargs)
    if legend:
        leg = ax.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)

    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.75**0.5)
    ax.set_xbound(lower=-0.01, upper=1.01)
    ax.set_ybound(lower=-0.01, upper=(0.75**0.5)+0.01)
    ax.axis('off')

    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    ax.triplot(triangle, c='k', lw=0.5)

    return fig, ax


def draw_func_contours(func, labels=None, nlevels=200, subdiv=5, fig=None,
                       ax=None, draw_lines=None, class_index=0, **kwargs):
    """
    Parameters:
    -----------
    labels: None, string or list of strings
        If labels == 'auto' it shows the class number on each corner
        If labels is a list of strings it shows each string in the
            corresponding corner
        If None does not show any label
    """
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)

    pvals = np.array([func(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)])

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    # FIXME I would like the following line to work, but the max value is
    # not shown. I had to do create manually the levels and increase the
    # max value by an epsilon. This could be a major problem if the epsilon
    # is not small for the original range of values
    # contour = ax.tricontourf(trimesh, pvals, nlevels, **kwargs)
    # contour = ax.tricontourf(trimesh, pvals, nlevels, extend='both')
    contour = ax.tricontourf(trimesh, pvals,
                             levels=np.linspace(pvals.min(), pvals.max()+1e-9,
                                                nlevels),
                             **kwargs)

    # Colorbar
    # TODO See if the following way to define the size of the bar can be used
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("bottom", size="5%", pad=0.1)
    # cb = fig.colorbar(contour, ax=cax, orientation='horizontal')
    cb = fig.colorbar(contour, ax=ax, orientation='horizontal',
                      fraction=0.05, pad=0.06)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    # cb.ax.xaxis.set_major_locator(ticker.AutoLocator())
    cb.update_ticks()

    if labels is None:
        labels = [r'$C_{}$'.format(i+1) for i in range(len(corners))]

    center = corners.mean(axis=0)
    for i, corner in enumerate(corners):
        text_x, text_y = corner - (center - corner)*0.1
        ax.text(text_x, text_y, labels[i], verticalalignment='center',
                horizontalalignment='center')

    if draw_lines is not None:
        lines = get_converging_lines(num_lines=draw_lines, mesh_precision=2,
                                     class_index=class_index)
        corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
        for line in lines:
            line = bc2xy(line, corners).T
            ax.plot(line[0], line[1])
            # l = mlines.Line2D()
            # ax.add_line(l)

    # Axes options
    ax.set_xlim(xmin=0, xmax=1)
    ax.set_ylim(ymin=0, ymax=0.75**0.5)
    ax.set_xbound(lower=0, upper=1)
    ax.set_ybound(lower=0, upper=0.75**0.5)
    ax.axis('equal')
    ax.axis('off')

    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    ax.triplot(triangle, c='k', lw=0.5)

    plt.gca().set_adjustable("box")
    return fig


def plot_converging_lines_pvalues(func, lines, i, ax):
    """
    Plots the probability values of the given function for each given line.
    The i indicates the class index from 0 to 2
    """
    # This orders the classes in the following manner:
    # C1, C2, C3
    # C2, C3, C1
    # C3, C1, C2
    classes = np.roll(np.array([0, 1, 2]), -i)

    for j, line in enumerate(lines):
        pvalues = np.array([func(p) for p in line]).flatten()
        ax.plot(line[:, i], pvalues,
                label=r'$C_{}/C_{} = {}/{}$'.format(
                    classes[1]+1, classes[2]+1, j, len(lines)-j-1))
    ax.legend()


def get_converging_lines(num_lines, mesh_precision=10, class_index=0,
                         tol=1e-6):
    """
    If class_index = 0
    Create isometric lines from the oposite side of C1 simplex to the C1 corner
    First line has C2 fixed to 0
    Last line has C3 fixed to 0
          Class 3  line 1 start
                 /\\
                /  \\
               /    \\ line 2 start
              /    - \\
             /   -/   \\
            /  -/      \\
           / -/      ---\\ line 3 start
          /-/  -----/    \\
         //---/           \\
        -------------------- line 4 start
    Class 1(lines end)      Class 2

    Else if class_index = [1, 2]
    Then the previusly described lines are rotated towards the indicated class.
    The lines always follow a clockwise order.
    """
    p = np.linspace(0, 1, mesh_precision).reshape(-1, 1)
    if num_lines == 1:
        q = [0.5]
    else:
        q = np.linspace(0, 1, num_lines).reshape(-1, 1)
    lines = [np.hstack((p, (1-p)*q[i], (1-p)*(1-q[i]))) for i in range(len(q))]
    if class_index > 0:
        indices = np.array([0, 1, 2])
        lines = [line[:, np.roll(indices, class_index)] for i, line in
                 enumerate(lines)]
    return np.clip(lines, tol, 1.0 - tol)
