# Configuration constants

# Default settling time before sweeps to avoid current spikes
DEFAULT_SETTLING_TIME_S = 3.0

# ML model input constraint: the coulomb blockade classifier model requires
# exactly 128 points per input window for inference
ML_MODEL_INPUT_SIZE = 128

# Use 2x peak spacing for initial multi-window sweep to ensure we capture
# full peak width plus sufficient context for accurate ML classification
INITIAL_WINDOW_MULTIPLIER = 2

PERTURBATION_DIVISOR = 20

# Use 80% of inter-peak distance for peak fitting windows to avoid
# overlapping windows while maximizing window size for better fit quality
# (empirically determined to balance resolution vs. overlap)
WINDOW_FRACTION = 0.8

# Default half-width (in points) for peak fitting windows when peaks are
# widely spaced. 128 points provides sufficient context for Lorentzian/sech²/
# Voigt fitting while avoiding edge effects
DEFAULT_WINDOW_HALF_WIDTH = 128

# Use 0.5x (half) of the initial step size for refined narrowed-range sweeps
# to improve peak center localization accuracy
REFINED_STEP_MULTIPLIER = 0.5


# Number of samples to average for each gate compensation measurement
NUM_OF_SAMPLES_FOR_AVERAGING = 5

# ML model constants
COULOMB_CLASSIFIER_MODEL = "coulomb-blockade-classifier-v3"
PEAK_DETECTOR_MODEL = "coulomb-blockade-peak-detector-v2"


MULTIPLER_OF_PEAK_SPACING = 0.3
