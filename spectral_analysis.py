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

    # Create a figure with 3 stacked panels.
    # The bottom panel (image) is half the height of each of the two graphs:
    # height ratios = [1, 1, 0.5]
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 1, 0.5]}
    )

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

    # 5. Bottom panel: show the image with no alpha, proportionally scaled
    # imshow will preserve the image aspect ratio by default; turn off axes.
    ax3.imshow(img_rgb)
    ax3.axis('off')  # hide ticks/labels so the image panel is clean

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