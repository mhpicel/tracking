from scipy import ndimage
import numpy as np


def get_vert_projection(grid, thresh=40):
    """Returns binary vertical projection from grid."""
    return np.any(grid > thresh, axis=0)


def get_filtered_frame(grid, min_size, thresh):
    """Returns a labeled frame from gridded radar data. Smaller objects are
    removed and the rest are labeled."""
    echo_height = get_vert_projection(grid, thresh)
    labeled_echo = ndimage.label(echo_height)[0]
    frame = clear_small_echoes(labeled_echo, min_size)
    return frame


def clear_small_echoes(label_image, min_size):
    """Takes in binary image and clears objects less than min_size."""
    flat_image = pd.Series(label_image.flatten())
    flat_image = flat_image[flat_image > 0]
    size_table = flat_image.value_counts(sort=False)
    small_objects = size_table.keys()[size_table < min_size]

    for obj in small_objects:
        label_image[label_image == obj] = 0
    label_image = ndimage.label(label_image)
    return label_image[0]


def extract_grid_data(grid_obj, field, grid_size, params):
    masked = grid_obj.fields[field]['data']
    masked.data[masked.data == masked.fill_value] = 0
    raw = masked.data[get_gs_alt(grid_size), :, :]
    frame = get_filtered_frame(masked.data, params['MIN_SIZE'],
                               params['FIELD_THRESH'])
    return raw, frame