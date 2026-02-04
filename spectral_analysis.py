import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import colorsys

def analyze_pixels_to_wavelengths(img_rgb, min_channel=10, max_channel=245, chroma_thresh=15, bin_size=1):
    """
    Convert image RGB pixels into estimated wavelengths and aggregate luminosity per wavelength bin.

    Steps:
    1. Flatten image into (N,3) RGB tuples.
    2. Filter out pixels where any channel <= min_channel or >= max_channel (too low / saturated),
       or where chroma (max-min) < chroma_thresh (nearly neutral gray).
    3. For remaining pixels, convert to HSV and map hue to an approximate wavelength.
    4. Compute pixel luminance and aggregate luminance into wavelength bins.

    Returns:
        bin_centers (np.ndarray): centers of wavelength bins (nm)
        luminosity_per_bin (np.ndarray): total luminance accumulated per bin
    """
    # Flatten to list of pixels
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)  # R,G,B per row

    # Filter: valid channel range (not near-zero and not saturated)
    valid_by_range = np.all((pixels > min_channel) & (pixels < max_channel), axis=1)

    # Filter: chroma (difference between max and min channel) must be above threshold
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    valid_by_chroma = chroma >= chroma_thresh

    valid_mask = valid_by_range & valid_by_chroma
    if not np.any(valid_mask):
        # No valid pixels
        bins = np.arange(380, 781, bin_size)
        centers = (bins[:-1] + bins[1:]) / 2.0
        return centers, np.zeros_like(centers)

    valid_pixels = pixels[valid_mask]

    # Compute luminance per pixel using Rec. 709 luma coefficients (on 0-255 scale)
    luminance = (0.2126 * valid_pixels[:, 0] +
                 0.7152 * valid_pixels[:, 1] +
                 0.0722 * valid_pixels[:, 2])

    # Convert RGB to HSV to get hue for each pixel. colorsys expects 0..1 inputs
    rgbs_norm = valid_pixels / 255.0

    # Map hue to wavelength with a piecewise linear approximation
    def hue_to_wavelength(h):
        # h is in [0.0, 1.0]
        # We'll map hue to the visible wavelength range 380..780 nm roughly
        # But the color -> wavelength mapping is not unique; this is a heuristic.
        # Define breakpoints (hue fractions) and corresponding wavelength ranges (nm)
        breakpoints = [0.0, 0.05, 0.17, 0.33, 0.50, 0.66, 0.82, 1.0]
        ranges = [(700, 620),  # deep red -> orange
                  (620, 590),  # orange -> yellow
                  (590, 530),  # yellow -> green
                  (530, 500),  # green -> cyan/teal
                  (500, 470),  # cyan -> blue
                  (470, 420),  # blue -> violet
                  (420, 380)]  # violet -> near-UV

        # Ensure h wrapped into [0,1)
        h = h % 1.0
        # Find segment
        for i in range(len(ranges)):
            lo = breakpoints[i]
            hi = breakpoints[i+1]
            if lo <= h <= hi or (i == len(ranges)-1 and h == 1.0):
                frac = 0.0 if hi == lo else (h - lo) / (hi - lo)
                wlo, whi = ranges[i]
                # linear interpolation within this range
                return wlo + frac * (whi - wlo)
        # fallback
        return 500.0

    # Compute wavelengths and collect luminance weights
    wavelengths = []
    for r, g, b in rgbs_norm:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        w = hue_to_wavelength(h)
        wavelengths.append(w)
    wavelengths = np.array(wavelengths)

    # Bin wavelengths and sum luminance per bin
    bins = np.arange(380, 781, bin_size)
    luminosity_per_bin, _ = np.histogram(wavelengths, bins=bins, weights=luminance)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    return bin_centers, luminosity_per_bin

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
    ax2 = fig.add_subplot(gs[1, 0])    # bottom-left: spectral distribution (will become line)
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

    # 4. Estimated Spectral Energy Distribution (now: wavelength vs luminosity line chart)
    # Use the new function to compute wavelength bins and aggregated luminosity.
    bin_centers, luminosity_per_bin = analyze_pixels_to_wavelengths(img_rgb)

    # Plot line chart (wavelength vs luminosity)
    ax2.plot(bin_centers, luminosity_per_bin, color='magenta', linewidth=1.0)
    ax2.set_title('Estimated Spectral Energy (Wavelength vs Luminosity)')
    ax2.set_xlabel('Approximate Wavelength (nm)')
    ax2.set_ylabel('Aggregated Pixel Luminance')
    ax2.set_xlim(bin_centers[0], bin_centers[-1])

    # 5. Thumbnail on the right of the second chart
    ax3.imshow(img_rgb)
    ax3.axis('off')  # hide ticks/labels so the thumbnail is clean

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
