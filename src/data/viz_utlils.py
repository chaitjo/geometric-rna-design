import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        elif key == "rmsds_list":
            print(f"{key}:")
            for pair, rmsd in value.items():
                print(f"\t{pair}, {rmsd}")
        else:
            print(f"{key}:\n\t{value}")


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
