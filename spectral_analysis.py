import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def wavelength_to_rgb(wavelength):
    """
    Approximate conversion from a wavelength in nm to an RGB tuple in [0, 1].
    Uses the commonly-used approximation (Dan Bruton's algorithm).
    Valid for ~380nm to 780nm. Values outside this range return (0,0,0).
    """
    wl = float(wavelength)
    if wl < 380 or wl > 780:
        return (0.0, 0.0, 0.0)

    if wl >= 380 and wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wl >= 440 and wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif wl >= 490 and wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif wl >= 510 and wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wl >= 580 and wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    else:  # 645nm - 780nm
        r = 1.0
        g = 0.0
        b = 0.0

    # Intensity correction near vision limits
    if wl >= 380 and wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif wl >= 420 and wl < 701:
        factor = 1.0
    elif wl >= 701 and wl <= 780:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    else:
        factor = 0.0

    gamma = 0.8  # a small gamma to make colors more perceptually linear here
    def apply(gval):
        if gval == 0.0:
            return 0.0
        return (gval * factor) ** gamma

    return (apply(r), apply(g), apply(b))

def pixels_to_wavelength_luminosity(
        img_rgb,
        wl_min=380,
        wl_max=780,
        n_wavelength_bins=70,
        min_channel=10,
        max_channel=245,
):
    """
    1) Convert image pixels to RGB tuples (done by reshaping).
    2) Filter out pixels where channel intensities are too low or too high
       (these pixels can't give reliable color ratios).
       - Default: exclude pixels with any channel < min_channel or > max_channel.
    3) For remaining pixels, compute normalized RGB ratios and assign each pixel to
       the wavelength (from a discretized set) whose predicted RGB-ratio best fits the pixel.
    4) Aggregate luminosity per wavelength (sum of luminance of assigned pixels).
    5) Return (wavelengths_array, luminosity_array)

    Notes:
    - Luminance per pixel uses the standard 0-255 weighted sum: Y = 0.299 R + 0.587 G + 0.114 B.
    - All channel thresholds and bin counts are configurable.
    """
    # Flatten pixels into Nx3 (R,G,B) float array
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)  # R,G,B order
    if pixels.size == 0:
        return np.array([]), np.array([])

    # Filter: remove pixels where any channel is too low or too high
    mask_valid = np.all(pixels >= min_channel, axis=1) & np.all(pixels <= max_channel, axis=1)
    valid_pixels = pixels[mask_valid]

    if valid_pixels.shape[0] == 0:
        # No valid pixels -> return wavelengths and zero luminosity
        wavelengths = np.linspace(wl_min, wl_max, n_wavelength_bins)
        return wavelengths, np.zeros_like(wavelengths)

    # Compute normalized color ratio (sum to 1) for matching; avoid divid by zero
    sums = valid_pixels.sum(axis=1, keepdims=True)
    # Exclude degenerate pixels with very low sum (should not happen due to min_channel)
    nonzero_mask = (sums[:, 0] > 0)
    valid_pixels = valid_pixels[nonzero_mask]
    sums = sums[nonzero_mask]

    ratios = valid_pixels / sums  # each row now [r/(r+g+b), g/(r+g+b), b/(r+g+b)]

    # Precompute model RGB ratios for each wavelength bin
    wavelengths = np.linspace(wl_min, wl_max, n_wavelength_bins)
    model_rgbs = np.array([wavelength_to_rgb(wl) for wl in wavelengths], dtype=np.float32)  # shape (M,3)

    # Normalize model RGBs to ratios (sum to 1). If a model is zero (out-of-range), replace with tiny epsilon.
    model_sums = model_rgbs.sum(axis=1, keepdims=True)
    model_sums[model_sums == 0] = 1e-9
    model_ratios = model_rgbs / model_sums  # shape (M,3)

    # For each pixel, find the model wavelength index with smallest squared error to pixel ratio
    # Vectorized distance: for N pixels, M models -> compute (N, M) distances
    # To avoid giant memory for very large N, we process in chunks
    N = ratios.shape[0]
    M = model_ratios.shape[0]
    chunk_size = 200000  # tuneable
    assigned_indices = np.empty(N, dtype=np.int32)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        chunk = ratios[start:end]  # (C,3)
        # compute squared distances to all model ratios -> (C, M)
        # using broadcasting efficiently
        diff = chunk[:, None, :] - model_ratios[None, :, :]  # (C,M,3)
        d2 = np.sum(diff * diff, axis=2)  # (C,M)
        assigned_indices[start:end] = np.argmin(d2, axis=1)

    # Compute luminance for valid (and non-degenerate) pixels using standard formula
    # Use original channel scale (0-255)
    # valid_pixels corresponds exactly to assigned_indices
    luminance = 0.299 * valid_pixels[:, 0] + 0.587 * valid_pixels[:, 1] + 0.114 * valid_pixels[:, 2]

    # Aggregate luminosity per wavelength bin
    luminosity_per_bin = np.bincount(assigned_indices, weights=luminance, minlength=M)
    # Ensure shape is (M,)
    if luminosity_per_bin.shape[0] < M:
        luminosity_per_bin = np.pad(luminosity_per_bin, (0, M - luminosity_per_bin.shape[0]))

    return wavelengths, luminosity_per_bin

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
    ax2 = fig.add_subplot(gs[1, 0])    # bottom-left: spectral distribution (replaced with line chart)
    ax3 = fig.add_subplot(gs[1, 1])    # bottom-right: thumbnail

    # 2. Define Approximate Sensor Peaks (Wavelengths in nm)
    # Most Bayer filters peak around these values
    peaks = {'Red': 650, 'Green': 530, 'Blue': 450}

    # 3. Calculate RGB Histograms (plot on ax1)
    colors = ('r', 'g', 'b')
    channel_names = ('Red', 'Green', 'Blue')

    # small positive floor to avoid log(0)
    eps_hist = 1e-1

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256]).flatten()
        # replace non-positive values with epsilon so log-scale can be used safely
        hist_safe = hist.copy()
        hist_safe[hist_safe <= 0] = eps_hist
        ax1.plot(hist_safe, color=col, label=f'{channel_names[i]} Channel')

    ax1.set_title('Pixel Intensity Distribution (Luminance)')
    ax1.set_xlabel('Brightness (0-255)')
    ax1.set_ylabel('Number of Pixels (log scale)')
    ax1.set_yscale('log')
    ax1.legend()

    # 4. Estimated Spectral Energy Distribution -> replaced with wavelength vs luminosity line chart
    # Use 70 ranges by default as requested
    wavelengths, luminosities = pixels_to_wavelength_luminosity(
        img_rgb,
        wl_min=380,
        wl_max=780,
        n_wavelength_bins=70,
        min_channel=10,
        max_channel=245,
    )

    # small positive floor to avoid log(0) on luminosity plot
    eps_lum = 1e-3

    # Plot line chart: wavelength (x) vs luminosity (y)
    if wavelengths.size > 0:
        lum_safe = luminosities.copy()
        lum_safe[lum_safe <= 0] = eps_lum
        ax2.plot(wavelengths, lum_safe, color='purple', linewidth=1.5)
        ax2.fill_between(wavelengths, lum_safe, color='purple', alpha=0.2)
        ax2.set_title('Estimated Spectral Luminosity (Wavelength vs Luminosity)')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Aggregated Luminance (sum, log scale)')
        ax2.set_yscale('log')
        ax2.grid(True, linestyle='--', alpha=0.4)
    else:
        ax2.text(0.5, 0.5, 'No valid pixels for wavelength estimation', ha='center', va='center')
        ax2.set_title('Estimated Spectral Luminosity (no data)')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Aggregated Luminance (sum)')

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