"""
Cell tracking code based on R scripts from Bhupendra Raut. This code uses the
fft method for image phase correlation to predict cell movement. The
Cell_tracks class near the end of the file employs the rest of the functions
and helper classes in this file.
"""

import copy
import datetime
import string
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
from scipy import ndimage, optimize

import pyart

# Settings for Tracking Method
LARGE_NUM = 1000
FIELD_THRESH = 32
ISO_THRESH = 8
MIN_SIZE = 32
NEAR_THRESH = 4
SEARCH_MARGIN = 8
FLOW_MARGIN = 20
MAX_DISPARITY = 999
MAX_FLOW_MAG = 50
MAX_SHIFT_DISP = 15


def print_params():
    """Prints tracking parameters."""
    print('FIELD_THRESH:', FIELD_THRESH)
    print('MIN_SIZE:', MIN_SIZE)
    print('SEARCH_MARGIN:', SEARCH_MARGIN)
    print('FLOW_MARGIN:', FLOW_MARGIN)
    print('MAX_DISPARITY:', MAX_DISPARITY)
    print('MAX_FLOW_MAG:', MAX_FLOW_MAG)
    print('MAX_SHIFT_DISP:', MAX_SHIFT_DISP)


def parse_grid_datetime(grid_obj):
    """Obtains datetime object from pyart grid_object."""
    date = grid_obj.time['units'][14:24]
    time = grid_obj.time['units'][25:-1]
    dt = datetime.datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M:%S')
    return dt


def get_vert_projection(grid, thresh=40):
    """Returns binary vertical projection from grid."""
    projection = np.empty_like(grid[0, :, :])
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            projection[i, j] = np.any(grid[:, i, j] > thresh)
    return projection


def get_grid_size(grid_obj):
    """calculates grid size per dimension given a grid object."""
    z_len = grid_obj.z['data'][-1] - grid_obj.z['data'][0]
    x_len = grid_obj.x['data'][-1] - grid_obj.x['data'][0]
    y_len = grid_obj.y['data'][-1] - grid_obj.y['data'][0]
    z_size = z_len / (grid_obj.z['data'].shape[0])
    x_size = x_len / (grid_obj.x['data'].shape[0])
    y_size = y_len / (grid_obj.y['data'].shape[0])
    return np.array([z_size, y_size, x_size])


def get_gs_alt(grid_size, alt_meters=1500):
    return np.int(np.round(alt_meters/grid_size[0]))


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


def extract_grid_data(grid_obj, field, grid_size):
    masked = grid_obj.fields[field]['data']
    masked.data[masked.data == masked.fill_value] = 0
    raw = masked.data[get_gs_alt(grid_size), :, :]
    frame = get_filtered_frame(masked.data, MIN_SIZE, FIELD_THRESH)
    return raw, frame


def get_pairs(image1, image2, global_shift, current_objects, record):
    """Given two images, this function identifies the matching objects and
    pairs them appropriately. See disparity function."""
    nobj1 = np.max(image1)
    nobj2 = np.max(image2)

    if nobj1 == 0:
        print('No echoes found in the first scan.')
        return
    elif nobj2 == 0:
        zero_pairs = np.zeros(nobj1)
        return zero_pairs, record

    obj_match, record = locate_allObjects(image1,
                                          image2,
                                          global_shift,
                                          current_objects,
                                          record)
    pairs = match_pairs(obj_match)
    return pairs, record


def match_pairs(obj_match):
    """Matches objects into pairs given a disparity matrix and removes bad
    matches. Bad matches have a disparity greater than the maximum
    threshold."""
    pairs = optimize.linear_sum_assignment(obj_match)

    for id1 in pairs[0]:
        if obj_match[id1, pairs[1][id1]] > MAX_DISPARITY:
            pairs[1][id1] = -1  # -1 indicates the object has died

    pairs = pairs[1] + 1  # ids in current_objects are 1-indexed
    return pairs


def locate_allObjects(image1, image2, global_shift, current_objects, record):
    """Matches all the objects in image1 to objects in image2. This is the main
    function called on a pair of images."""
    nobj1 = np.max(image1)
    nobj2 = np.max(image2)

    if (nobj2 == 0) or (nobj1 == 0):
        print('No echoes to track!')
        return

    obj_match = np.full((nobj1, np.max((nobj1, nobj2))),
                        LARGE_NUM, dtype='f')

    for obj_id1 in np.arange(nobj1) + 1:
        obj1_extent = get_objExtent(image1, obj_id1)
        shift = get_ambient_flow(obj1_extent, image1, image2, FLOW_MARGIN)
        if shift is None:
            record.count_case(5)
            shift = global_shift

        shift, record = correct_shift(shift, current_objects, obj_id1,
                                      global_shift, record)

        search_box = predict_searchExtent(obj1_extent, shift, SEARCH_MARGIN)
        search_box = check_search_box(search_box, image2.shape)
        obj_found = find_objects(search_box, image2)
        disparity = get_disparity_all(obj_found, image2,
                                      search_box, obj1_extent)
        obj_match = save_obj_match(obj_id1, obj_found, disparity, obj_match)

    return obj_match, record


def correct_shift(pc_shift, current_objects, obj_id1, global_shift, record):
    """Takes in flow vector based on local phase correlation (see
    get_std_flow) and compares it to the last headings of the object and
    the global_shift vector for that timestep. Corrects accordingly.
    Note: At the time of this function call, current_objects has not yet been
    updated for the current frame1 and frame2, so the id2s in current_objects
    correspond to the objects in the current frame1."""

    if current_objects is None:
        last_heads = ()
    else:
        obj_index = current_objects['id2'] == obj_id1
        last_heads = current_objects['last_heads'][obj_index].flatten()
        last_heads = np.round(last_heads * record.interval_ratio, 2)

    if len(last_heads) == 0:
        if shifts_disagree(pc_shift, global_shift, record, MAX_SHIFT_DISP):
            case = 0
            corrected_shift = global_shift
        else:
            case = 1
            corrected_shift = (pc_shift + global_shift)/2

    elif shifts_disagree(pc_shift, last_heads, record, MAX_SHIFT_DISP):
        if shifts_disagree(pc_shift, global_shift, record, MAX_SHIFT_DISP):
            case = 2
            corrected_shift = last_heads
        else:
            case = 3
            corrected_shift = pc_shift

    else:
        case = 4
        corrected_shift = (pc_shift + last_heads)/2

    corrected_shift = np.round(corrected_shift, 2)

    record.count_case(case)
    record.record_shift(corrected_shift, global_shift,
                        last_heads, pc_shift, case)
    return corrected_shift, record


def shifts_disagree(shift1, shift2, record, thresh):
    shift1 = shift1*record.grid_size[1:]
    shift2 = shift2*record.grid_size[1:]
    shift_disparity = euclidean_dist(shift1, shift2)
    return shift_disparity/record.interval.seconds > thresh


def get_objExtent(labeled_image, obj_label):
    """Takes in labeled image and finds the radius, area, and center of the
    given object."""
    obj_index = np.argwhere(labeled_image == obj_label)

    xlength = np.max(obj_index[:, 0]) - np.min(obj_index[:, 0]) + 1
    ylength = np.max(obj_index[:, 1]) - np.min(obj_index[:, 1]) + 1
    obj_radius = np.max((xlength, ylength))/2
    obj_center = np.round(np.median(obj_index, axis=0), 0)
    obj_area = len(obj_index[:, 0])

    obj_extent = {'obj_center': obj_center, 'obj_radius': obj_radius,
                  'obj_area': obj_area, 'obj_index': obj_index}
    return obj_extent


def get_ambient_flow(obj_extent, img1, img2, margin):
    """Takes in object extent and two images and returns ambient flow. Margin
    is the additional region around the object used to compute the flow
    vectors."""
    row_lb = obj_extent['obj_center'][0] - obj_extent['obj_radius'] - margin
    row_ub = obj_extent['obj_center'][0] + obj_extent['obj_radius'] + margin
    col_lb = obj_extent['obj_center'][1] - obj_extent['obj_radius'] - margin
    col_ub = obj_extent['obj_center'][1] + obj_extent['obj_radius'] + margin
    row_lb = np.int(row_lb)
    row_ub = np.int(row_ub)
    col_lb = np.int(col_lb)
    col_ub = np.int(col_ub)

    dims = img1.shape
    if row_lb <= 0 or col_lb <= 0 or row_ub > dims[0] or col_ub > dims[1]:
        return None

    flow_region1 = img1[row_lb:row_ub+1, col_lb:col_ub+1]
    flow_region2 = img2[row_lb:row_ub+1, col_lb:col_ub+1]
    return fft_flowvectors(flow_region1, flow_region2)


def fft_flowvectors(im1, im2):
    """Estimates flow vectors in two images using cross covariance."""
    if (np.max(im1) == 0) or (np.max(im2) == 0):
        return None

    crosscov = fft_crosscov(im1, im2)
    sigma = (1/8) * min(crosscov.shape)
    cov_smooth = ndimage.filters.gaussian_filter(crosscov, sigma)
    dims = np.array(im1.shape)
    pshift = np.argwhere(cov_smooth == np.max(cov_smooth))[0]
    pshift = (pshift+1) - np.round(dims/2, 0)
    return pshift


def fft_crosscov(im1, im2):
    """Computes cross correlation matrix using FFT method."""
    fft1_conj = np.conj(np.fft.fft2(im1))
    fft2 = np.fft.fft2(im2)
    normalize = abs(fft2*fft1_conj)
    normalize[normalize == 0] = 1  # prevent divide by zero error
    cross_power_spectrum = (fft2*fft1_conj)/normalize
    crosscov = np.fft.ifft2(cross_power_spectrum)
    crosscov = np.real(crosscov)
    return fft_shift(crosscov)


def fft_shift(fft_mat):
    """Rearranges the cross correlation matrix so that 'zero' frequency or DC
    component is in the middle of the matrix. Taken from stackoverflow Que.
    30630632."""
    if type(fft_mat) is np.ndarray:
        rd2 = np.int(fft_mat.shape[0]/2)
        cd2 = np.int(fft_mat.shape[1]/2)
        quad1 = fft_mat[:rd2, :cd2]
        quad2 = fft_mat[:rd2, cd2:]
        quad3 = fft_mat[rd2:, cd2:]
        quad4 = fft_mat[rd2:, :cd2]
        centered_t = np.concatenate((quad4, quad1), axis=0)
        centered_b = np.concatenate((quad3, quad2), axis=0)
        centered = np.concatenate((centered_b, centered_t), axis=1)
        return centered
    else:
        print('input to fft_shift() should be a matrix')
        return


def get_global_shift(im1, im2, magnitude):
    """Returns standardazied global shift vector. im1 and im2 are full frames
    of raw DBZ values."""
    if im2 is None:
        return None
    shift = fft_flowvectors(im1, im2)
    shift[shift > magnitude] = magnitude
    shift[shift < -magnitude] = -magnitude
    return shift


def predict_searchExtent(obj1_extent, shift, search_radius):
    """Predicts search extent/region for the object in image2 given the image
    shift."""
    shifted_center = obj1_extent['obj_center'] + shift

    x1 = shifted_center[0] - search_radius
    x2 = shifted_center[0] + search_radius + 1
    y1 = shifted_center[1] - search_radius
    y2 = shifted_center[1] + search_radius + 1
    x1 = np.int(x1)
    x2 = np.int(x2)
    y1 = np.int(y1)
    y2 = np.int(y2)

    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2,
            'center_pred': shifted_center, 'valid': True}


def check_search_box(search_box, img_dims):
    """Checks if search_box is within the boundaries of the frame. Clips to
    edges of frame if out of bounds. Marks as invalid if too small."""
    if search_box['x1'] < 0:
        search_box['x1'] = 0
    if search_box['y1'] < 0:
        search_box['y1'] = 0
    if search_box['x2'] > img_dims[0]:
        search_box['x2'] = img_dims[0]
    if search_box['y2'] > img_dims[1]:
        search_box['y2'] = img_dims[1]
    if ((search_box['x2'] - search_box['x1'] < 5) or
            (search_box['y2'] - search_box['y1'] < 5)):
        search_box['valid'] = False
    return search_box


def find_objects(search_box, image2):
    """Identifies objects found in the search region."""
    if not search_box['valid']:
        obj_found = np.array(-1)
    else:
        search_area = image2[search_box['x1']:search_box['x2'],
                             search_box['y1']:search_box['y2']]
        obj_found = np.unique(search_area)

    return obj_found


def get_disparity_all(obj_found, image2, search_box, obj1_extent):
    """Returns disparities of all objects found within the search box."""
    if np.max(obj_found) <= 0:
        disparity = np.array([LARGE_NUM])
    else:
        obj_found = obj_found[obj_found > 0]
        disparity = get_disparity(obj_found, image2,
                                  search_box, obj1_extent)
#        obj_found = obj_found[obj_found > 0]
#        if len(obj_found) == 1:
#            disparity = get_disparity(obj_found, image2,
#                                      search_box, obj1_extent)
#            if disparity <= 3:
#                disparity = np.array([0])
#        else:
#            disparity = get_disparity(obj_found, image2,
#                                      search_box, obj1_extent)
    return disparity


def save_obj_match(obj_id1, obj_found, disparity, obj_match):
    """Saves disparity values in obj_match matrix. If disparity is greater than
    MAX_DISPARITY, saves a large number."""
    disparity[disparity > MAX_DISPARITY] = LARGE_NUM
    if np.max(obj_found) > 0:
        obj_found = obj_found[obj_found > 0]
        obj_found = obj_found - 1
        obj_id1 = obj_id1 - 1
        obj_match[obj_id1, obj_found] = disparity
    return obj_match


def get_disparity(obj_found, image2, search_box, obj1_extent):
    """Computes disparities for objects in obj_found."""
    dist_pred = np.empty(0)
    # dist_actual = np.empty(0)
    change = np.empty(0)
    for target_obj in obj_found:
        target_extent = get_objExtent(image2, target_obj)
        euc_dist = euclidean_dist(target_extent['obj_center'],
                                  search_box['center_pred'])
        dist_pred = np.append(dist_pred, euc_dist)

        # euc_dist = euclidean_dist(target_extent['obj_center'],
        #                           obj1_extent['obj_center'])
        # dist_actual = np.append(dist_actual, euc_dist)

        size_changed = get_sizeChange(target_extent['obj_area'],
                                      obj1_extent['obj_area'])
        change = np.append(change, size_changed)

    disparity = dist_pred + change
    return disparity


def euclidean_dist(vec1, vec2):
    """Computes euclidean distance."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.sqrt(sum((vec1-vec2)**2))
    return dist


def get_sizeChange(size1, size2):
    """Returns change in size of an echo as the ratio of the larger size to the
    smaller, minus 1."""
    if (size1 < 5) and (size2 < 5):
        return 0
    elif size1 >= size2:
        return size1/size2 - 1
    else:
        return size2/size1 - 1


def write_tracks(old_tracks, record, current_objects, obj_props):
    """Writes x and y grid position to tracks dataframe for each object present
    in scan."""
    print('Writing tracks for scan', record.scan)

    nobj = len(obj_props['id1'])
    scan_num = [record.scan] * nobj
    uid = current_objects['uid']

    new_tracks = pd.DataFrame({
        'scan': scan_num,
        'uid': uid,
        'time': record.time,
        'grid_x': obj_props['grid_x'],
        'grid_y': obj_props['grid_y'],
        'lon': obj_props['lon'],
        'lat': obj_props['lat'],
        'area': obj_props['area'],
        'vol': obj_props['volume'],
        'max': obj_props['field_max'],
        'max_alt': obj_props['max_height'],
        'isolated': obj_props['isolated']
    })
    new_tracks.set_index(['scan', 'uid'], inplace=True)
    tracks = old_tracks.append(new_tracks)
    return tracks


def check_merging(dead_obj_id1, current_objects, obj_props):
    """Checks if a dead object has merged into another object. The criterion
    for merging is based on the euclidean distance between object centers. This
    method does not perfrom very well and will be improved."""
    nobj_frame1 = len(current_objects['id1'])
    c_dist_all = []
    checked_id1 = []

    if all(current_objects['id2'] == 0):
        return 0

    for check_obj in range(nobj_frame1):
        if current_objects['id2'][check_obj] != 0:
            dead_xy = (obj_props['center'][dead_obj_id1][0],
                       obj_props['center'][dead_obj_id1][1])
            merge_xy = (obj_props['center'][check_obj][0],
                        obj_props['center'][check_obj][1])
            c_dist = euclidean_dist(merge_xy, dead_xy)
            if c_dist < np.sqrt(obj_props['area'][check_obj]):
                c_dist_all.append(c_dist)
                checked_id1.append(check_obj)

    if len(c_dist_all) == 0:
        return 0
    else:
        product_id1 = checked_id1[np.argmin(c_dist_all)]
        return current_objects['uid'][product_id1]


def check_isolation(raw, filtered):
    nobj = np.max(filtered)
    iso_filtered = get_filtered_frame(raw, MIN_SIZE, ISO_THRESH)
    nobj_iso = np.max(iso_filtered)
    iso = np.empty(nobj, dtype='bool')

    for iso_id in np.arange(nobj_iso) + 1:
        objects = np.unique(filtered[np.where(iso_filtered == iso_id)])
        objects = objects[objects != 0]
        if len(objects) == 1:
            iso[objects - 1] = True
        else:
            iso[objects - 1] = False
    return iso


# def check_isolation(raw, filtered):
#    nobj = np.max(filtered)
#    iso_filtered = get_filtered_frame(raw, MIN_SIZE, ISO_THRESH)
#    nobj_iso = np.max(iso_filtered)
#    iso = np.empty(nobj, dtype='bool')
#
#    for iso_id in np.arange(nobj_iso) + 1:
#        objects = filtered[np.where(iso_filtered == iso_id)]
#        ids = np.unique(objects)
#        ids = ids[ids != 0]
#        if len(ids) == 1:
#            if np.sum(objects == 0)/len(objects) < 0.7:
#                iso[ids - 1] = True
#        else:
#            iso[ids - 1] = False
#    return iso


def init_uids(first_frame, second_frame, pairs, counter):
    """Returns a dictionary for objects with unique ids and their corresponding
    ids in frame1 and frame1. This function is called when echoes are detected
    after a period of no echoes."""
    nobj = np.max(first_frame)

    id1 = np.arange(nobj) + 1
    uid = counter.next_uid(count=nobj)
    id2 = pairs
    obs_num = np.zeros(nobj, dtype='i')
    origin = np.array(['-1']*nobj)

    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_last_heads(first_frame, second_frame,
                                        current_objects)
    return current_objects, counter


def attach_last_heads(frame1, frame2, current_objects):
    """attaches last heading information to current_objects dictionary."""
    nobj = len(current_objects['uid'])
    heads = np.ma.empty((nobj, 2))
    for obj in range(nobj):
        if ((current_objects['id1'][obj] > 0) and
                (current_objects['id2'][obj] > 0)):
            center1 = get_object_center(current_objects['id1'][obj], frame1)
            center2 = get_object_center(current_objects['id2'][obj], frame2)
            heads[obj, :] = center2 - center1
        else:
            heads[obj, :] = np.ma.array([-999, -999], mask=[True, True])

    current_objects['last_heads'] = heads
    return current_objects


def get_object_center(obj_id, labeled_image):
    """Returns index of center pixel of the given object id from labeled image.
    The center is calculated as the median pixel of the object extent; it is
    not a true centroid."""
    obj_index = np.argwhere(labeled_image == obj_id)
    center = np.median(obj_index, axis=0).astype('i')
    return center


def update_current_objects(frame1, frame2, pairs, old_objects, counter):
    """Removes dead objects, updates living objects, and assigns new uids to
    new-born objects."""
    nobj = np.max(frame1)
    id1 = np.arange(nobj) + 1
    uid = np.array([], dtype='str')
    obs_num = np.array([], dtype='i')
    origin = np.array([], dtype='str')

    for obj in np.arange(nobj) + 1:
        if obj in old_objects['id2']:
            obj_index = old_objects['id2'] == obj
            uid = np.append(uid, old_objects['uid'][obj_index])
            obs_num = np.append(obs_num, old_objects['obs_num'][obj_index] + 1)
            origin = np.append(origin, old_objects['origin'][obj_index])
        else:
            #  obj_orig = get_origin_uid(obj, frame1, old_objects)
            obj_orig = '-1'
            origin = np.append(origin, obj_orig)
            if obj_orig != '-1':
                uid = np.append(uid, counter.next_cid(obj_orig))
            else:
                uid = np.append(uid, counter.next_uid())
            obs_num = np.append(obs_num, 0)

    id2 = pairs
    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_last_heads(frame1, frame2, current_objects)
    return current_objects, counter


def get_origin_uid(obj, frame1, old_objects):
    """Returns the unique id of the origin for a given object. Retunrs '-1' if
    the object has not split off from another object."""
    origin_id = find_origin(obj, frame1)
    if origin_id is None:
        return '-1'

    if origin_id not in old_objects['id2']:
        return '-1'

    origin_index = np.argwhere(old_objects['id2'] == origin_id)[0][0]

    origin_uid = old_objects['uid'][origin_index]
    return origin_uid


def find_origin(id1_new, frame1):
    """This function checks near by objects in the frame for the given new-born
    object. Returns uid of an object that existed before the new born object,
    has a comparable or larger size, and is within a predefined distance.
    Returns '-1' if no such object exists."""
    if np.max(frame1) == 1:
        return None

    object_ind = np.argwhere(frame1 == id1_new)
    object_size = object_ind.shape[0]

    neighbour_ind = np.argwhere((frame1 > 0) & (frame1 != id1_new))
    neighbour_size = neighbour_ind.shape[0]

    neighbour_dist = np.array([])
    neighbour_id = np.array([])
    size_ratio = np.array([])
    size_diff = np.array([])

    for pix in range(object_size):
        for neighbour in range(neighbour_size):
            euc_dist = euclidean_dist(object_ind[pix, :],
                                      neighbour_ind[neighbour, :])
            neighbour_dist = np.append(neighbour_dist, euc_dist)
            pix_id = neighbour_ind[neighbour, :]
            neighbour_id = np.append(neighbour_id,
                                     frame1[pix_id[0], pix_id[1]])

    nearest_object_id = neighbour_id[neighbour_dist < NEAR_THRESH]
    # the_nearest_object = neighbour_id[neighbour_dist == min(neighbour_dist)]

    if len(nearest_object_id) == 0:
        return None

    neigh_objects = np.unique(nearest_object_id)
    for object in neigh_objects:
        nearest_object_size = len(frame1[frame1 == object])
        size_ratio = np.append(size_ratio, nearest_object_size/object_size)
        size_diff = np.append(size_diff, nearest_object_size - object_size)

    big_ratio_obj = neigh_objects[size_ratio == max(size_ratio)]
    big_diff_obj = neigh_objects[size_diff == max(size_diff)]

    if big_ratio_obj == big_diff_obj:
        return big_diff_obj[0]
    else:
        return big_diff_obj[0]


def get_objectProp(image1, grid1, field, record):
    """Returns dictionary of object properties for all objects found in
    image1."""
    id1 = []
    center = []
    grid_x = []
    grid_y = []
    area = []
    longitude = []
    latitude = []
    field_max = []
    max_height = []
    volume = []
    nobj = np.max(image1)

    unit_dim = record.grid_size
    unit_alt = unit_dim[0]/1000
    unit_area = (unit_dim[1]*unit_dim[2])/(1000**2)
    unit_vol = (unit_dim[0]*unit_dim[1]*unit_dim[2])/(1000**3)

    raw3D = grid1.fields[field]['data'].data

    for obj in np.arange(nobj) + 1:
        obj_index = np.argwhere(image1 == obj)
        id1.append(obj)

        # 2D frame stats
        center.append(np.median(obj_index, axis=0))
        this_centroid = np.round(np.mean(obj_index, axis=0), 3)
        grid_x.append(this_centroid[1])
        grid_y.append(this_centroid[0])
        area.append(obj_index.shape[0] * unit_area)

        rounded = np.round(this_centroid).astype('i')
        lon = grid1.get_point_longitude_latitude()[0][rounded[0], rounded[1]]
        lat = grid1.get_point_longitude_latitude()[1][rounded[0], rounded[1]]
        longitude.append(np.round(lon, 4))
        latitude.append(np.round(lat, 4))

        # raw 3D grid stats

        obj_slices = [raw3D[:, ind[0], ind[1]] for ind in obj_index]
        field_max.append(np.max(obj_slices))
        filtered_slices = [obj_slice > FIELD_THRESH
                           for obj_slice in obj_slices]
        heights = [np.arange(raw3D.shape[0])[ind] for ind in filtered_slices]
        max_height.append(np.max(np.concatenate(heights)) * unit_alt)
        volume.append(np.sum(filtered_slices) * unit_vol)

    # cell isolation
    isolation = check_isolation(raw3D, image1)

    objprop = {'id1': id1,
               'center': center,
               'grid_x': grid_x,
               'grid_y': grid_y,
               'area': area,
               'field_max': field_max,
               'max_height': max_height,
               'volume': volume,
               'lon': longitude,
               'lat': latitude,
               'isolated': isolation}
    return objprop


class Counter(object):
    """This is a helper class for the get_tracks method in the Cell_tracks
    class. Counter objects generate and keep track of unique cell ids. This
    class will further developed when merging and splitting functionality is
    improved."""

    def __init__(self):
        """uid is an integer that tracks the number of independently formed
        cells. The cid dictionary keeps track of 'children' --i.e., cells that
        have split off from another cell."""
        self.uid = -1
        self.cid = {}

    def next_uid(self, count=1):
        """Incremented for every new independently formed cell."""
        new_uids = self.uid + np.arange(count) + 1
        self.uid += count
        return np.array([str(uid) for uid in new_uids])

    def next_cid(self, pid):
        """Returns parent uid with appended letter to denote child."""
        if pid in self.cid.keys():
            self.cid[pid] += 1
        else:
            self.cid[pid] = 0
        letter = string.ascii_lowercase[self.cid[pid]]
        return pid + letter


class Record(object):
    """Record objects keep track of shift correction at each timestep. They
    also hold information about the time of each scan."""
    def __init__(self, grid_obj):
        self.scan = -1
        self.time = None
        self.interval = None
        self.interval_ratio = None
        self.grid_size = get_grid_size(grid_obj)
        self.shifts = pd.DataFrame()
        self.new_shifts = pd.DataFrame()
        self.correction_tally = {'case0': 0, 'case1': 0, 'case2': 0,
                                 'case3': 0, 'case4': 0, 'case5': 0}

    def count_case(self, case_num):
        """Updates correction_tally dictionary. This is used to monitor the
        shift correction process."""
        self.correction_tally['case' + str(case_num)] += 1

    # Shift Correction Case Guide:
    # case0 - new object, local_shift and global_shift disagree, returns global
    # case1 - new object, returns local_shift
    # case2 - local disagrees with last head and global, returns last head
    # case3 - local disagrees with last head, returns local
    # case4 - local and last head agree, returns average of both
    # case5 - flow regions empty or at edge of frame, returns global_shift

    def record_shift(self, corr, gl_shift, l_heads, pc_shift, case):
        """Records corrected shift, phase shift, global shift, and last heads
        per object per timestep. This information can be used to monitor and
        refine the shift correction algorithm in the correct_shift function."""
        if len(l_heads) == 0:
            l_heads = np.ma.array([-999, -999], mask=[True, True])

        new_shift_record = pd.DataFrame()
        new_shift_record['scan'] = [self.scan]
        new_shift_record['uid'] = ['uid']
        new_shift_record['corrected'] = [corr]
        new_shift_record['global'] = [gl_shift]
        new_shift_record['last_heads'] = [l_heads]
        new_shift_record['phase'] = [pc_shift]
        new_shift_record['case'] = [case]

        self.new_shifts = self.new_shifts.append(new_shift_record)

    def add_uids(self, current_objects):
        """Because of the chronology of the get_tracks process, object uids
        cannot be added to the shift record at the time of correction, so they
        must be added later in the process."""
        if len(self.new_shifts) > 0:
            self.new_shifts['uid'] = current_objects['uid']
            self.new_shifts.set_index(['scan', 'uid'], inplace=True)
            self.shifts = self.shifts.append(self.new_shifts)
            self.new_shifts = pd.DataFrame()

    def update_scan_and_time(self, grid_obj1, grid_obj2=None):
        """Updates the scan number and associated time. This information is
        used for obtaining object properties as well as for the interval ratio
        correction of last_heads vectors."""
        self.scan += 1
        self.time = parse_grid_datetime(grid_obj1)
        if grid_obj2 is None:
            # tracks for last scan are being written
            return
        time2 = parse_grid_datetime(grid_obj2)
        old_diff = self.interval
        self.interval = time2 - self.time
        if old_diff is not None:
            self.interval_ratio = self.interval.seconds/old_diff.seconds


class Cell_tracks(object):
    """This is the main class in the module. It allows tracks
    objects to be built using lists of pyart grid objects."""

    def __init__(self, field='reflectivity'):
        self.__params = None
        self.field = field
#        self.grids = []
        self.last_grid = None
        self.counter = None
        self.record = None
        self.current_objects = None
        self.tracks = pd.DataFrame()

        self.__saved_record = None
        self.__saved_counter = None
        self.__saved_objects = None

    def __save_params(self):
        self.__params = {'FIELD_THRESH': FIELD_THRESH,
                         'MIN_SIZE': MIN_SIZE,
                         'SEARCH_MARGIN': SEARCH_MARGIN,
                         'FLOW_MARGIN': FLOW_MARGIN,
                         'MAX_FLOW_MAG': MAX_FLOW_MAG,
                         'MAX_DISPARITY': MAX_DISPARITY,
                         'MAX_SHIFT_DISP': MAX_SHIFT_DISP}

    def print_params(self):
        """prints tracking parameters"""
        if self.__params is None:
            print('this object is empty')
        else:
            print('tracking paramters used for this object')
            for key, val in self.__params.items():
                print(key + ':', val)
            print('\n')
        return

    def __save(self):
        """Saves deep copies of record, counter, and current_objects."""
        self.__saved_record = copy.deepcopy(self.record)
        self.__saved_counter = copy.deepcopy(self.counter)
        self.__saved_objects = copy.deepcopy(self.current_objects)

    def __load(self):
        """Loads saved copies of record, counter, and current_objects. If new
        tracks are appended to existing tracks via the get_tracks method, the
        most recent scan prior to the addition must be overwritten to link up
        with the new scans. Because of this, record, counter and
        current_objects must be reverted to their state in the penultimate
        iteration of the loop in get_tracks. See get_tracks for details."""
        self.record = self.__saved_record
        self.counter = self.__saved_counter
        self.current_objects = self.__saved_objects

    def get_tracks(self, grids):
        """Obtains tracks given a list of pyart grid objects. This is the
        primary method of the tracks class. This method makes use of all of the
        functions and helper classes defined above."""
        start_time = datetime.datetime.now()
        self.__save_params()

        if self.record is None:
            # tracks object being initialized
#            start_scan = 0
#            grid_obj2 = grids[0]
            grid_obj2 = next(grids)
            self.grid_size = get_grid_size(grid_obj2)
            self.counter = Counter()
            self.record = Record(grid_obj2)
        else:
            # tracks object being updated
#            start_scan = -1
#            grid_obj2 = self.grids[-1]
            grid_obj2 = self.last_grid
            self.tracks.drop(self.record.scan + 1)  # last scan is overwritten
#        self.grids += grids
#        end_scan = len(grids)

        if self.current_objects is None:
            newRain = True
        else:
            newRain = False

#        print('Total scans in this list:', len(grids))

        raw2, frame2 = extract_grid_data(grid_obj2, self.field, self.grid_size)

#        for scan in range(start_scan + 1, end_scan + 1):
        while grid_obj2 is not None:
            grid_obj1 = grid_obj2
            raw1 = raw2
            frame1 = frame2

            try:
                grid_obj2 = next(grids)
            except StopIteration:
                grid_obj2 = None

#            if scan != end_scan:
            if grid_obj2 is not None:
#                grid_obj2 = grids[scan]
                self.record.update_scan_and_time(grid_obj1, grid_obj2)
                raw2, frame2 = extract_grid_data(grid_obj2,
                                                 self.field,
                                                 self.grid_size)
            else:
                # setup to write final scan
                self.__save()
                self.last_grid = grid_obj1
                self.record.update_scan_and_time(grid_obj1)
                raw2 = None
                frame2 = np.zeros_like(frame1)

            if np.max(frame1) == 0:
                newRain = True
                print('No cells found in scan', self.record.scan)
                self.current_objects = None
                continue

            global_shift = get_global_shift(raw1, raw2, MAX_FLOW_MAG)
            pairs, record = get_pairs(frame1,
                                      frame2,
                                      global_shift,
                                      self.current_objects,
                                      self.record)

            if newRain:
                # first nonempty scan after a period of empty scans
                self.current_objects, self.counter = init_uids(
                    frame1,
                    frame2,
                    pairs,
                    self.counter
                )
                newRain = False
            else:
                self.current_objects, self.counter = update_current_objects(
                    frame1,
                    frame2,
                    pairs,
                    self.current_objects,
                    self.counter
                )

            obj_props = get_objectProp(frame1, grid_obj1, self.field, record)
            self.record.add_uids(self.current_objects)
            self.tracks = write_tracks(self.tracks, self.record,
                                       self.current_objects, obj_props)
            # scan loop end
        self.__load()
        time_elapsed = datetime.datetime.now() - start_time
        print('\n')
        print('time elapsed', np.round(time_elapsed.seconds/60, 1), 'minutes')
        return

#    def animate(self, outfile_name, arrows=False, isolation=False, fps=1):
#        """Creates gif animation of tracked cells."""
#        grid_size = get_grid_size(self.grids[0])
#
#        def animate_frame(nframe):
#            """ Animate a single frame of gridded reflectivity including
#            uids. """
#            plt.clf()
#            print("Frame:", nframe)
#            grid = self.grids[nframe]
#            display = pyart.graph.GridMapDisplay(grid)
#            ax = fig_grid.add_subplot(111)
#            display.plot_basemap()
#            display.plot_grid(self.field, level=2, vmin=-8, vmax=64,
#                              mask_outside=False, cmap=pyart.graph.cm.NWSRef)
##            display.plot_grid(self.field, level=2, vmin=0, vmax=2,
##                              mask_outside=False)
#
#            if nframe in self.tracks.index.levels[0]:
#                frame_tracks = self.tracks.loc[nframe]
#                for ind, uid in enumerate(frame_tracks.index):
#
#                    if isolation and not frame_tracks['isolated'].iloc[ind]:
#                        continue
#
#                    x = frame_tracks['grid_x'].iloc[ind]*grid_size[2]
#                    y = frame_tracks['grid_y'].iloc[ind]*grid_size[1]
#                    ax.annotate(uid, (x, y), fontsize=20)
#                    if arrows and ((nframe, uid) in self.record.shifts.index):
#                        shift = self.record.shifts \
#                            .loc[nframe, uid]['corrected']
#                        shift = shift * grid_size[1:]
#                        ax.arrow(x, y, shift[1], shift[0],
#                                 head_width=3*grid_size[1],
#                                 head_length=6*grid_size[1])
#            del grid
#            return
#        fig_grid = plt.figure(figsize=(10, 8))
#        anim_grid = animation.FuncAnimation(fig_grid, animate_frame,
#                                            frames=len(self.grids))
#        anim_grid.save(outfile_name,
#                       writer='imagemagick', fps=fps)
