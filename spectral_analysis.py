import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def analyze_light_frequency(image_path):
    # 1. Load the image
    # Note: OpenCV loads as BGR by default
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: Could not load image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Base layout proportions (top row full width, bottom row split)
    # We will compute the thumbnail width fraction (f) so that:
    #   0.2 <= f <= 0.5
    # and the image at width = f * total_width fits within the bottom-row height
    top_bottom_height_ratios = [1.0, 0.6]  # relative heights for top and bottom rows

    # Create a temporary figure to get size/dpi for pixel-based calculations
    fig = plt.figure(figsize=(12, 8))
    dpi = fig.dpi
    fig_w_px = fig.get_size_inches()[0] * dpi
    fig_h_px = fig.get_size_inches()[1] * dpi

    # Image aspect ratio (width / height) in pixels
    img_h, img_w = img_rgb.shape[0], img_rgb.shape[1]
    img_aspect = img_w / img_h if img_h > 0 else 1.0

    # Compute bottom row height in pixels (based on the height ratios)
    total_height_ratio = sum(top_bottom_height_ratios)
    bottom_row_height_fraction = top_bottom_height_ratios[1] / total_height_ratio
    bottom_row_h_px = fig_h_px * bottom_row_height_fraction

    # Compute maximum thumbnail width fraction allowed by bottom-row height
    # For a thumbnail width of f * fig_w_px, the displayed image height would be:
    #   displayed_h_px = (f * fig_w_px) / img_aspect
    # We require displayed_h_px <= bottom_row_h_px  -> f <= bottom_row_h_px * img_aspect / fig_w_px
    f_max_by_height = (bottom_row_h_px * img_aspect) / fig_w_px

    # Constrain fraction to [0.2, 0.5] and pick the largest feasible value within those bounds
    thumbnail_frac = min(0.5, max(0.2, f_max_by_height))

    # Convert fractions to width_ratios for GridSpec.
    # If left fraction = (1 - thumbnail_frac), right fraction = thumbnail_frac
    left_frac = max(0.0, 1.0 - thumbnail_frac)
    width_ratios = [left_frac, thumbnail_frac]

    # Create GridSpec with computed width ratios
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=top_bottom_height_ratios,
        width_ratios=width_ratios,
        hspace=0.25,
        wspace=0.15,
    )

    ax1 = fig.add_subplot(gs[0, :])    # top row spanning both columns
    ax2 = fig.add_subplot(gs[1, 0])    # bottom-left: spectral distribution
    ax3 = fig.add_subplot(gs[1, 1])    # bottom-right: thumbnail

    # 2. Define Approximate Sensor Peaks (Wavelengths in nm)
    # Most Bayer filters peak around these values
    peaks = {'Red': 650, 'Green': 530, 'Blue': 450}

    # 3. Calculate RGB Histograms (plot on ax1)
    colors = ('r', 'g', 'b')
    channel_names = ('Red', 'Green', 'Blue')

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        ax1.plot(hist, color=col, label=f'{channel_names[i]} Channel')

    ax1.set_title('Pixel Intensity Distribution (Luminance)')
    ax1.set_xlabel('Brightness (0-255)')
    ax1.set_ylabel('Number of Pixels')
    ax1.legend()

    # 4. Estimated Spectral Energy Distribution (plot on ax2)
    channel_totals = [np.sum(img_rgb[:, :, 0]), np.sum(img_rgb[:, :, 1]), np.sum(img_rgb[:, :, 2])]
    wavelengths = [peaks['Red'], peaks['Green'], peaks['Blue']]

    sorted_data = sorted(zip(wavelengths, channel_totals, ['red', 'green', 'blue']))
    w_vals, i_vals, c_vals = zip(*sorted_data)

    ax2.bar(w_vals, i_vals, width=30, color=c_vals, alpha=0.6, edgecolor='black')
    ax2.set_title('Estimated Spectral Energy Distribution')
    ax2.set_xlabel('Approximate Wavelength (nm)')
    ax2.set_ylabel('Total Accumulated Intensity')
    ax2.set_xticks([450, 530, 650])
    ax2.set_xticklabels(['450nm (Blue)', '530nm (Green)', '650nm (Red)'])

    # 5. Thumbnail on the right of the second chart
    # Show the image while preserving aspect ratio. The GridSpec allocation ensures
    # the thumbnail area is no less than 1/5 and no more than 1/2 of the bottom-row width,
    # and we computed the fraction so that the image can fill that space as large as possible.
    ax3.imshow(img_rgb)
    ax3.axis('off')  # hide ticks/labels so the thumbnail is clean

    # Optional: annotate the chosen fraction (helpful for debugging or UI feedback)
    # ax3.set_title(f'Thumbnail ({thumbnail_frac*100:.0f}% width)')

    plt.tight_layout()
    # Set window title if backend supports it (safe-guarded)
    try:
        plt.get_current_fig_manager().set_window_title(f"{image_path}: Histograms + Image")
    except Exception:
        pass
    plt.show()

if len(sys.argv) == 1:
    print(f"Usage: {sys.argv[0]} file1.jpg file2.jpg ...")
else:
    args = sys.argv[1:]  # Get all arguments except the script name
    for arg in args:
        analyze_light_frequency(arg)