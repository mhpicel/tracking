import datetime
import os
import string

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
from scipy import ndimage, optimize

import pyart


def get_grid(file):
    """Returns gridded reflectivity values from radar file."""
    radar = pyart.io.read(file)
    grid = pyart.map.grid_from_radars(
        radar, grid_shape=(34, 240, 240),
        grid_limits=((0, 17000), (-60000, 60000), (-60000, 60000)),
        fields=['reflectivity'], gridding_algo="map_gates_to_grid",
        weighting_function='BARNES')
    return grid.fields['reflectivity']['data']


def get_vert_projection(grid, thresh=40):
    """Returns binary vertical projection from grid."""
    projection = np.empty_like(grid[0, :, :])
    for i in range(grid.shape[1]):
        for j in range(grid.shape[2]):
            projection[i, j] = np.any(grid[:, i, j] > thresh)
    return projection


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


def change_base_epoch(time_seconds, from_epoch):
    """Changes base epoch of time_second from from_epoch to global variable
    BASE_EPOCH."""
    to_epoch = BASE_EPOCH
    epoch_diff = from_epoch - to_epoch
    epoch_diff_seconds = epoch_diff.days * 86400
    time_new = time_seconds + epoch_diff_seconds
    return time_new


def get_pairs(image1, image2, global_shift):
    """Given two images, this function identifies the matching objects and
    pairs them appropriately. See disparity function."""
    nobj1 = np.max(image1)
    nobj2 = np.max(image2)

    if nobj1 == 0:
        print('No echoes found in the first scan.')
        return
    elif nobj2 == 0:
        zero_pairs = np.zeros(nobj1)
        return zero_pairs

    obj_match = locate_allObjects(image1, image2, global_shift)
    pairs = match_pairs(obj_match)
    return pairs


def match_pairs(obj_match):
    """Matches objects into pairs and removes bad matches. Bad matches have
    a disparity greater than the maximum threshold."""
    pairs = optimize.linear_sum_assignment(obj_match)

    for id1 in pairs[0]:
        if obj_match[id1, pairs[1][id1]] > MAX_DISPARITY:
            pairs[1][id1] = -1  # -1 indicates the object has died

    pairs = pairs[1] + 1  # ids in current_objects are 1-indexed
    return pairs


def locate_allObjects(image1, image2, global_shift):
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
        shift = get_std_flow(obj1_extent, image1, image2,
                             FLOW_MARGIN, STD_FLOW_MAG)
        if shift is None:
            global correction_record
            correction_record['case4'] += 1
            shift = global_shift

        if 'current_objects' in globals():
            shift = correct_shift(shift, current_objects, obj_id1,
                                  global_shift)
        else:
            shift = global_shift

        search_box = predict_searchExtent(obj1_extent, shift, SEARCH_MARGIN)
        search_box = check_search_box(search_box, image2.shape)
        obj_found = find_objects(search_box, image2)
        disparity = get_disparity_all(obj_found, image2,
                                      search_box, obj1_extent)
        obj_match = save_obj_match(obj_id1, obj_found, disparity, obj_match)

    return obj_match


def correct_shift(this_shift, current_objects, object_id1, global_shift):
    """Takes in flow vector based on local phase correlation (see
    get_std_flow) and compares it to the last headings of the object and
    the global_shift vector for that timestep. Corrects accordingly.
    Note: At the time of this function call, current_objects has not yet been
    updated for the current frame1 and frame2, so the id2s in current_objects
    correspond to the objects in the current frame1."""
    obj_index = current_objects['id2'] == object_id1
    last_heads = np.ma.append(current_objects['xhead'][obj_index],
                              current_objects['yhead'][obj_index])

    global shift_record
    new_record = pd.DataFrame({'scan': [np.int(scan - 1)],
                               'uid': ['new'],
                               'shift_x': [this_shift[0]],
                               'shift_y': [this_shift[1]],
                               'head_x': [-999],
                               'head_y': [-999],
                               'global_x': [global_shift[0]],
                               'global_y': [global_shift[1]]})
    if len(last_heads > 0):
        new_record['head_x'] = [last_heads[0]]
        new_record['head_y'] = [last_heads[1]]
        new_record['uid'] = current_objects['uid'][obj_index]
    shift_record = shift_record.append(new_record)

    global correction_record
    if len(last_heads) == 0:
        if any(abs(this_shift - global_shift) > MAX_SHIFT_DISP):
            correction_record['case0'] += 1
            return global_shift
        correction_record['case1'] += 1
        return (this_shift + global_shift)/2

    elif any(abs(this_shift - last_heads) > MAX_SHIFT_DISP):
        if any(abs(this_shift - global_shift) > MAX_SHIFT_DISP):
            correction_record['case2'] += 1
            return last_heads
        correction_record['case3'] += 1
        return this_shift

    else:
        correction_record['case5'] += 1
        return (this_shift+last_heads)/2


def get_objExtent(labeled_image, obj_label):
    """Takes in labeled image and finds the radius, area, and center of the
    given object."""
    obj_index = np.argwhere(labeled_image == obj_label)

    xlength = np.max(obj_index[:, 0]) - np.min(obj_index[:, 0]) + 1
    ylength = np.max(obj_index[:, 1]) - np.min(obj_index[:, 1]) + 1
    obj_radius = np.max((xlength, ylength))/2
    obj_center = (np.min(obj_index[:, 0]) + obj_radius,
                  np.min(obj_index[:, 1]) + obj_radius)
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


def get_std_flow(obj_extent, img1, img2, margin, magnitude):
    """Alternative to get_ambient_flow. Flow vector's magnitude is clipped to
    given magnitude."""
    shift = get_ambient_flow(obj_extent, img1, img2, margin)
    if shift is None:
        return None

    shift[shift > magnitude] = magnitude
    shift[shift < -magnitude] = -magnitude
    return shift


def fft_flowvectors(im1, im2):
    """Estimates flow vectors in two images using cross covariance."""
    if (np.max(im1) == 0) or (np.max(im2) == 0):
        return None

    crosscov = fft_crosscov(im1, im2)
    sigma = (1/8) * min(crosscov.shape)
    cov_smooth = ndimage.filters.gaussian_filter(crosscov, sigma)
    dims = im1.shape
    pshift = np.argwhere(cov_smooth == np.max(cov_smooth))[0]
    pshift = (pshift+1) - (dims[0]/2)
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
        if len(obj_found) == 1:
            disparity = get_disparity(obj_found, image2,
                                      search_box, obj1_extent)
            if disparity <= 3:
                disparity = np.array([0])
        else:
            disparity = get_disparity(obj_found, image2,
                                      search_box, obj1_extent)
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


def write_tracks(old_tracks, scan, current_objects, obj_props):
    """Writes x and y grid position to tracks dataframe for each object present
    in scan."""
    print('writing track', scan)

    nobj = len(obj_props['id1'])
    scan_num = [scan-1] * nobj
    x_pos = obj_props['x']
    y_pos = obj_props['y']
    uid = current_objects['uid']

    new_tracks = pd.DataFrame({'scan': scan_num, 'x': x_pos,
                               'y': y_pos, 'uid': uid})
    tracks = old_tracks.append(new_tracks)
    print(new_tracks)
    return tracks

# def write_duration(outNC, current_objects):
#    nobj = len(current_objects['id1'])
#
#    for obj in range(nobj):
#        if current_objects['id2'][obj] == 0:
#            uid = current_objects['uid'][obj]
#            outNC.variables['duration'][uid] = current_objects['obs_num'][obj]
#            outNC.variables['origin'][uid] = current_objects['origin'][obj]
#            merged_in = check_merging(obj, current_objects, obj_props)
#            outNC.variables['merged'][uid] = merged_in


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
            dead_xy = (obj_props['x'][dead_obj_id1],
                       obj_props['y'][dead_obj_id1])
            merge_xy = (obj_props['x'][check_obj],
                        obj_props['y'][check_obj])
            c_dist = euclidean_dist(merge_xy, dead_xy)
            if c_dist < np.sqrt(obj_props['area'][check_obj]):
                c_dist_all.append(c_dist)
                checked_id1.append(check_obj)

    if len(c_dist_all) == 0:
        return 0
    else:
        product_id1 = checked_id1[np.argmin(c_dist_all)]
        return current_objects['uid'][product_id1]


# def write_survival(outNC, survival_stat, time, scan):
#    outNC.variables['survival'][:5, scan] = survival_stat
#    outNC.variables['time_utc'][scan] = time


def init_uids(first_frame, second_frame, pairs):
    """Returns a dictionary for objects with unique ids and their corresponding
    ids in frame1 and frame1. This function is called when echoes are detected
    after a period of no echoes."""
    nobj = np.max(first_frame)

    id1 = np.arange(nobj) + 1
    uid = np.array(next_uid(count=nobj))
    id2 = pairs
    obs_num = np.zeros(nobj, dtype='i')
    origin = np.array(['-1']*nobj)

    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_xyheads(first_frame, second_frame,
                                     current_objects)
    return current_objects


def attach_xyheads(frame1, frame2, current_objects):
    """attaches last heading information to current_objects dictionary."""
    nobj = len(current_objects['uid'])
    xhead = np.ma.empty(0)
    yhead = np.ma.empty(0)
    na_array = np.ma.array([-999], mask=[True])

    for obj in range(nobj):
        if ((current_objects['id1'][obj] > 0) and
                (current_objects['id2'][obj] > 0)):
            center1 = get_object_center(current_objects['id1'][obj], frame1)
            center2 = get_object_center(current_objects['id2'][obj], frame2)
            xhead = np.ma.append(xhead, center2[0] - center1[0])
            yhead = np.ma.append(yhead, center2[1] - center1[1])
        else:
            xhead = np.ma.append(xhead, na_array)
            yhead = np.ma.append(yhead, na_array)

    current_objects['xhead'] = xhead
    current_objects['yhead'] = yhead
    return current_objects


def get_object_center(obj_id, labeled_image):
    """Returns index of center pixel of the given object id from labeled image.
    The center is calculated as the median pixel of the object extent; it is
    not a true centroid."""
    obj_index = np.argwhere(labeled_image == obj_id)
    center_x = np.int(np.median(obj_index[:, 0]))
    center_y = np.int(np.median(obj_index[:, 1]))
    return np.array((center_x, center_y))


def update_current_objects(frame1, frame2, pairs, old_objects):
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
            obj_orig = get_origin_uid(obj, frame1, old_objects)
            origin = np.append(origin, obj_orig)
            if obj_orig != '-1':
                uid = np.append(uid, next_cid(obj_orig))
            else:
                uid = np.append(uid, next_uid())
            obs_num = np.append(obs_num, 0)

    id2 = pairs
    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_xyheads(frame1, frame2, current_objects)
    return current_objects


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


def next_uid(count=1):
    """Returns sequence of next unique ids and increments uid_counter."""
    global uid_counter
    this_uid = uid_counter + np.arange(count) + 1
    uid_counter = uid_counter + count
    return [str(uid) for uid in this_uid]


def next_cid(uid):
    """Returns next child id given the uid of a parent object. The cid is a
    lowercase ascii character that will be appended to the uid of the parent
    object to form the uid of the child."""
    global cid_record
    if uid in cid_record.keys():
        cid_record[uid] += 1
        letter = string.ascii_lowercase[cid_record[uid]]
        return uid + letter
    else:
        cid_record[uid] = 0
        letter = string.ascii_lowercase[cid_record[uid]]
        return uid + letter


def get_objectProp(image1):
    """Calculates object properties. Returns size and location."""
    id1 = np.empty(0, dtype='i')
    x = np.empty(0, dtype='i')
    y = np.empty(0, dtype='i')
    area = np.empty(0, dtype='i')
    nobj = np.max(image1)

    for obj in np.arange(nobj) + 1:
        obj_index = np.argwhere(image1 == obj)
        id1 = np.append(id1, obj)
        x = np.append(x, np.int(np.median(obj_index[:, 0])))
        y = np.append(y, np.int(np.median(obj_index[:, 1])))
        area = np.append(area, len(obj_index[:, 1]))

    objprop = {'id1': id1, 'x': x, 'y': y, 'area': area}
    return objprop


def survival_stats(pairs, num_obj2):
    """Returns a list with number of objects survived, died, and born between
    this step and the next one."""
    obj_lived = len(pairs[pairs > 0])
    obj_died = len(pairs) - obj_lived
    obj_born = num_obj2 - obj_lived
    return [obj_lived, obj_died, obj_born, num_obj2]

# _____________________________________________________________________________
# _______________________settings for tracking method__________________________
START_TIME = datetime.datetime.now()
BASE_EPOCH = datetime.datetime(1970, 1, 1)
DBZ_THRESH = 24
MIN_SIZE = 50
MAX_SHIFT_DISP = 8
NEAR_THRESH = 4
SEARCH_MARGIN = 10
FLOW_MARGIN = 20
STD_FLOW_MAG = 20
LARGE_NUM = 1000
MAX_DISPARITY = 35
# _____________________________________________________________________________
# _____________________read radar files and generate tracks____________________
# get files from local directory
local_dir = '/home/mhpicel/Practice/csapr data/sur/20110525/'
local_dir_data = [local_dir + ext for ext in os.listdir(local_dir)]
local_dir_data.sort()
file_list = local_dir_data

# initialize counters
uid_counter = -1
cid_record = {}

correction_record = {'case0': 0, 'case1': 0, 'case2': 0,
                     'case3': 0, 'case4': 0, 'case5': 0}
# Shift Correction Guide:
# case0 - new object, local_shift and global_shift disagree, returns global
# case1 - new object, returns local_shift
# case2 - local disagrees with last head and global, returns last head
# case3 - local disagrees with last head, returns local
# case4 - flow regions empty or at edge of frame, returns global_shift
# case5 - local and last head agree, returns average of both

shift_record = pd.DataFrame({'shift_x': [],
                             'shift_y': [],
                             'head_x': np.ma.array([]),
                             'head_y': np.ma.array([])})

start_scan = 10
# end_scan = len(file_list)-1
end_scan = 25

newRain = True
if 'current_objects' in globals():
    del current_objects

print('Total scans in this file', end_scan - start_scan + 1)

grid2 = get_grid(local_dir_data[start_scan])
raw2 = grid2[3, :, :]
frame2 = get_filtered_frame(grid2, MIN_SIZE, DBZ_THRESH)
tracks = pd.DataFrame()

for scan in range(start_scan + 1, end_scan + 1):
    grid1 = grid2
    raw1 = raw2
    frame1 = frame2

    grid2 = get_grid(local_dir_data[scan])
    raw2 = grid2[3, :, :]
    frame2 = get_filtered_frame(grid2, MIN_SIZE, DBZ_THRESH)

    if scan == end_scan:
        frame2 = np.zeros_like(frame2)

    if np.max(frame1) == 0:
        newRain = True
        print('newRain')
        if 'current_objects' in globals():
            del current_objects
        continue

    global_shift = get_global_shift(raw1, raw2, STD_FLOW_MAG)
    pairs = get_pairs(frame1, frame2, global_shift)
    obj_props = get_objectProp(frame1)

    if newRain:
        current_objects = init_uids(frame1, frame2, pairs)
        newRain = False
    else:
        current_objects = update_current_objects(frame1, frame2,
                                                 pairs, current_objects)

    tracks = write_tracks(tracks, scan, current_objects, obj_props)
    # scan loop end


shift_record.set_index(['scan', 'uid'], inplace=True)
shift_record.sort_index(inplace=True)
time_elapsed = datetime.datetime.now() - START_TIME
print('\n')
print('closing files')
print('time elapsed', np.round(time_elapsed.seconds/60), 'minutes')
# _____________________________________________________________________________
# ___________________________generate animation________________________________
anim_filename = 'tracks_maint.gif'


def animate_grid(nframe):
    """ Animate a single frame of gridded reflectivity including uids. """
    plt.clf()
    nscan = nframe + start_scan
    fname = local_dir_data[nscan]
    print("Frame:", nframe, "File:", fname)

    radar = pyart.io.read(fname)
    grid = pyart.map.grid_from_radars(
        radar, grid_shape=(34, 240, 240),
        grid_limits=((0, 17000), (-60000, 60000), (-60000, 60000)),
        fields=['reflectivity'], gridding_algo="map_gates_to_grid",
        weighting_function='BARNES')
    display = pyart.graph.GridMapDisplay(grid)
    ax = fig_grid.add_subplot(111)
    display.plot_basemap(lat_lines=np.arange(30, 46, .2),
                         lon_lines=np.arange(-110, -75, .4))
    display.plot_grid('reflectivity', level=3, vmin=-8, vmax=64,
                      cmap=pyart.graph.cm.NWSRef)

    frame_tracks = tracks[tracks['scan'] == nscan]
    for ind, uid in enumerate(frame_tracks['uid']):
        x = (frame_tracks['x'].iloc[ind])*500
        y = (frame_tracks['y'].iloc[ind])*500
        ax.annotate(uid, (y, x), fontsize=20)
    return

fig_grid = plt.figure(figsize=(10, 8))
anim_grid = animation.FuncAnimation(fig_grid, animate_grid,
                                    frames=(end_scan - start_scan))
anim_grid.save(anim_filename,
               writer='imagemagick', fps=1)
