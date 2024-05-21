import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import draw_rna.draw as d
from draw_rna.draw_utils import seq2col

from lovely_numpy import lo
import lovely_tensors as lt
lt.monkey_patch()


def print_rna_data(data):
    for key, value in data.items():
        if key == "sasa_list":
            print(f"{key}:")
            for sasa in value:
                print(f"\t{lo(sasa)}")
        elif key == "coords_list":
            print(f"{key}:")
            for coords in value:
                print(f"\t{lo(coords)}")
        elif key == "sec_struct_list":
            print(f"{key}:")
            for sec_struct in value:
                print(f"\t{sec_struct}")
        elif key == "eterna_sec_struct_list":
            print(f"{key}:")
            for sec_struct in value:
                print(f"\t{sec_struct}")
        elif key == "rmsds_list":
            print(f"{key}:")
            for pair, rmsd in value.items():
                print(f"\t{pair}, {rmsd}")
        else:
            print(f"{key}:\n\t{value}")


def draw_2d_struct(seq, secstruct, c=None, line=False, large_mode=False, cmap='viridis', rotation=0, vmin=None, vmax=None, alpha=None, ax=None):
    '''
    Draw sequence with secondary structure.
    
    Inputs:
    c (string or array-like).  If string, characters must correspond to colors.
    If array-like obj, used as mapping for colormap (provided in cmap), or a string.
    line (bool): draw secstruct as line.
    large_mode: draw outer loop as straight line.
    rotation: rotate molecule (in degrees).

    Source: https://github.com/DasLab/draw_rna
    '''
    if c is not None:
        assert len(c) == len(seq)
        if isinstance(c[0], float):
            d.draw_rna(seq, secstruct, c, line=line, ext_color_file=True, cmap_name = cmap, vmin=vmin, vmax=vmax,
             rotation=rotation, large_mode = large_mode, alpha=alpha, ax=ax)
        else:
            d.draw_rna(seq, secstruct, c,  line=line, cmap_name=cmap, large_mode=large_mode, vmin=vmin, vmax=vmax,
             rotation=rotation, alpha=alpha, ax=ax)

    else:
        d.draw_rna(seq, secstruct, seq2col(seq), line=line, cmap_name = cmap, vmin=vmin, vmax=vmax,
         large_mode = large_mode, rotation=rotation, alpha=alpha, ax=ax)

    if ax is None:
        plt.show()


def plot_multiple_3d_point_clouds(point_clouds):
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define a list of colors for each point cloud
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, point_cloud in enumerate(point_clouds):
        # Extract the X, Y, and Z coordinates from the point cloud
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]

        # Plot the 3D points with a unique color
        ax.scatter(x, y, z, c=colors[i % len(colors)], marker='o', label=f'Point Cloud {i+1}')

        # Join the points with a line
        ax.plot(x, y, z, c=colors[i % len(colors)])

    # Set labels for the axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Add a legend to distinguish the point clouds
    ax.legend()

    # Move legend to the upper right corner and outside of the plot
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # Display the plot
    plt.show()
