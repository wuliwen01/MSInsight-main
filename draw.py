from obspy.imaging.beachball import beachball
import matplotlib.pyplot as plt
import numpy as np


def draw_maxbrightness(data, title, save=False):
    """Draw three cross-sections at the maximum brightness value of a 3D matrix
    """
    # Find the index of the maximum brightness value
    max_idx = np.unravel_index(np.argmax(data), data.shape)
    # print("Maximum position:", max_idx, "Maximum value:", data[max_idx])

    # Extract three cross-sections
    # Left view (YZ plane) - along the X axis
    left_view = data[max_idx[0], :, :]

    # Top view (XY plane) - along the Z axis
    top_view = data[:, :, max_idx[2]]

    # Front view (XZ plane) - along the Y axis
    front_view = data[:, max_idx[1], :]

    # Set font to Times New Roman
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })

    # Create 2x2 subplots, leaving one empty
    fig, axs = plt.subplots(2, 2, figsize=(4, 4), dpi=400)
    # fig.suptitle(title, fontsize=16)

    # Left view
    ax1 = axs[0, 0]
    im1 = ax1.imshow(left_view, cmap='gnuplot',
                     aspect='equal', origin='lower', vmin=0)
    ax1.set_title('YZ Plane')
    ax1.set_xticks([])  # Remove x-axis ticks
    ax1.set_yticks([])  # Remove y-axis ticks
    # plt.colorbar(im1, ax=ax1)

    # Top view
    ax2 = axs[0, 1]
    im2 = ax2.imshow(top_view.T, cmap='gnuplot',
                     aspect='equal', origin='lower', vmin=0)
    ax2.set_title('XY Plane')
    ax2.set_xticks([])  # Remove x-axis ticks
    ax2.set_yticks([])  # Remove y-axis ticks
    # plt.colorbar(im2, ax=ax2)

    # Front view
    ax3 = axs[1, 1]
    im3 = ax3.imshow(front_view.T, cmap='gnuplot',
                     aspect='equal', origin='lower', vmin=0)
    ax3.set_title('XZ Plane')
    ax3.set_xticks([])  # Remove x-axis ticks
    ax3.set_yticks([])  # Remove y-axis ticks
    # plt.colorbar(im3, ax=ax3)

    # Empty subplot (can be deleted, but usually better to keep it)
    axs[1, 0].axis('off')

    # Adjust layout
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Display image
    if save:
        plt.savefig(f"maxbrightness_{title}.png")
    else:
        plt.show()
    return max_idx


def draw_2stage_maxbrightness(s1, s2, s2_gr, title):
    """Draw three cross-sections at the maximum brightness value of two stages

    Args:
        s1 (ndarray): First stage brightness field(x,y,z)
        s2 (ndarray): Second stage brightness field(x,y,z)
        s2_gr (dict): Grid point range of s2 brightness field
    """
    max_idx_s2 = np.array(np.unravel_index(np.argmax(s2), s2.shape))
    max_idx_s1 = max_idx_s2 // 2
    max_s2 = s2[max_idx_s2[0], max_idx_s2[1], max_idx_s2[2]]
    # Extract three cross-sections
    # Left view (YZ plane) - along the X axis
    left_view_s1 = s1[max_idx_s1[0], :, :]
    left_view_s2 = s2[max_idx_s2[0], :, :]
    # Front view (XZ plane) - along the Y axis
    front_view_s1 = s1[:, max_idx_s1[1], :]
    front_view_s2 = s2[:, max_idx_s2[1], :]
    # Top view (XY plane) - along the Z axis
    top_view_s1 = s1[:, :, max_idx_s1[2]]
    top_view_s2 = s2[:, :, max_idx_s2[2]]

    # Create 2x2 subplots, leaving one empty
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    fig.suptitle(title, fontsize=16)
    # Left view
    ax1 = axs[0, 0]
    ax1.imshow(left_view_s1, cmap='gnuplot', alpha=0.5,
               aspect='equal', origin='lower', vmin=0, vmax=max_s2)
    ax1.imshow(left_view_s2, cmap='gnuplot', aspect='equal', origin='lower',
               extent=[s2_gr['z_min']/2, s2_gr['z_max']/2, s2_gr['y_min']/2, s2_gr['y_max']/2], vmin=0, vmax=max_s2)
    ax1.set_xlim([-0.5, left_view_s1.shape[1]-0.5])
    ax1.set_ylim([-0.5, left_view_s1.shape[0]-0.5])
    ax1.set_title('Left View (YZ Plane)')
    # plt.colorbar(im1s2, ax=ax1)

    # Top view
    ax2 = axs[0, 1]
    ax2.imshow(top_view_s1.T, cmap='gnuplot', alpha=0.5,
               aspect='equal', origin='lower', vmin=0, vmax=max_s2)
    ax2.imshow(top_view_s2.T, cmap='gnuplot', aspect='equal', origin='lower',
               extent=[s2_gr['x_min']/2, s2_gr['x_max']/2, s2_gr['y_min']/2, s2_gr['y_max']/2], vmin=0, vmax=max_s2)
    ax2.set_xlim([-0.5, top_view_s1.shape[1]-0.5])
    ax2.set_ylim([-0.5, top_view_s1.shape[0]-0.5])
    ax2.set_title('Top View (XY Plane)')
    # plt.colorbar(im2s2, ax=ax2)

    # Front view
    ax3 = axs[1, 1]
    ax3.imshow(front_view_s1.T, cmap='gnuplot', alpha=0.5,
               aspect='equal', origin='lower', vmin=0, vmax=max_s2)
    ax3.imshow(front_view_s2.T, cmap='gnuplot', aspect='equal', origin='lower',
               extent=[s2_gr['x_min']/2, s2_gr['x_max']/2, s2_gr['z_min']/2, s2_gr['z_max']/2], vmin=0, vmax=max_s2)
    ax3.set_xlim([-0.5, front_view_s1.shape[0]-0.5])
    ax3.set_ylim([-0.5, front_view_s1.shape[1]-0.5])
    ax3.set_title('Front View (XZ Plane)')
    # plt.colorbar(im3s2, ax=ax3)

    # Empty subplot (can be deleted, but usually better to keep it)
    axs[1, 0].axis('off')

    plt.show()


def draw_beachball(strike, dip, rake, size=200, linewidth=1):
    """
    Plot a beachball diagram using given focal mechanism parameters.

    :param source: Dictionary containing focal mechanism parameters.
    :param size: Size of the beachball diagram.
    :param linewidth: Line width of the beachball diagram.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(3, 3))

    # Plot the beachball
    beachball([strike, dip, rake], size=size,
              linewidth=linewidth, facecolor='black', bgcolor='white', fig=fig)
