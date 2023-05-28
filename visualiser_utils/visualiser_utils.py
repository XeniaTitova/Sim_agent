from matplotlib.patches import Rectangle
import numpy as np


def drawRectangle(ax, xy_cent, ang, w, h, color='red'):
    xy = [xy_cent[0] - w / 2, xy_cent[1] - h / 2]
    rect = Rectangle(xy, w, h, angle=ang, color=color, rotation_point='center')
    ax.add_patch(rect)


def drawSegment(ax, xy, dir, color='black', linewidth=2, linestyle='dashed'):
    x = [xy[0], xy[0] + dir[0]]
    y = [xy[1], xy[1] + dir[1]]
    ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle)
