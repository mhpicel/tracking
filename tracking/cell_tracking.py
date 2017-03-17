import datetime
import string
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
from scipy import ndimage, optimize

import pyart

# Settings for Tracking Method
BASE_EPOCH = datetime.datetime(1970, 1, 1)
LARGE_NUM = 1000
DBZ_THRESH = 24
MIN_SIZE = 50
NEAR_THRESH = 4
SEARCH_MARGIN = 8
FLOW_MARGIN = 20
MAX_DISPARITY = 25
MAX_FLOW_MAG = 20
MAX_SHIFT_DISP = 8


def print_params():
    """Prints tracking parameters."""
    print('DBZ_THRESH:', DBZ_THRESH)
    print('MIN_SIZE:', MIN_SIZE)
    print('SEARCH_MARGIN:', SEARCH_MARGIN)
    print('FLOW_MARGIN:', FLOW_MARGIN)
    print('MAX_DISPARITY:', MAX_DISPARITY)
    print('MAX_FLOW_MAG:', MAX_FLOW_MAG)
    print('MAX_SHIFT_DISP:', MAX_SHIFT_DISP)


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


# def change_base_epoch(time_seconds, from_epoch):
#    """Changes base epoch of time_second from from_epoch to global variable
#    BASE_EPOCH."""
#    to_epoch = BASE_EPOCH
#    epoch_diff = from_epoch - to_epoch
#    epoch_diff_seconds = epoch_diff.days * 86400
#    time_new = time_seconds + epoch_diff_seconds
#    return time_new


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
    """Matches objects into pairs and removes bad matches. Bad matches have
    a disparity greater than the maximum threshold."""
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
        shift = get_std_flow(obj1_extent, image1, image2,
                             FLOW_MARGIN, MAX_FLOW_MAG)
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
        last_heads = np.ma.append(current_objects['xhead'][obj_index],
                                  current_objects['yhead'][obj_index])

    if len(last_heads) == 0:
        if any(abs(pc_shift - global_shift) > MAX_SHIFT_DISP):
            case = 0
            corrected_shift = global_shift
        else:
            case = 1
            corrected_shift = (pc_shift + global_shift)/2

    elif any(abs(pc_shift - last_heads) > MAX_SHIFT_DISP):
        if any(abs(pc_shift - global_shift) > MAX_SHIFT_DISP):
            case = 2
            corrected_shift = last_heads
        else:
            case = 3
            corrected_shift = pc_shift

    else:
        case = 4
        corrected_shift = (pc_shift + last_heads)/2

    record.count_case(case)
    record.record_shift(corrected_shift, global_shift,
                        last_heads, pc_shift, case)
    return corrected_shift, record


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
    print('Writing tracks for scan', scan - 1)

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
    current_objects = attach_xyheads(first_frame, second_frame,
                                     current_objects)
    return current_objects, counter


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
            obj_orig = get_origin_uid(obj, frame1, old_objects)
            origin = np.append(origin, obj_orig)
            if obj_orig != '-1':
                uid = np.append(uid, counter.next_cid(obj_orig))
            else:
                uid = np.append(uid, counter.next_uid())
            obs_num = np.append(obs_num, 0)

    id2 = pairs
    current_objects = {'id1': id1, 'uid': uid, 'id2': id2,
                       'obs_num': obs_num, 'origin': origin}
    current_objects = attach_xyheads(frame1, frame2, current_objects)
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


# def new_find_origin(id1_new, frame1):
#    object_ind = np.argwhere(frame1 == id1_new)
#


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


class Counter(object):

    def __init__(self):
        self.uid = -1
        self.cid = {}

    def next_uid(self, count=1):
        new_uids = self.uid + np.arange(count) + 1
        self.uid += count
        return np.array([str(uid) for uid in new_uids])

    def next_cid(self, pid):
        if pid in self.cid.keys():
            self.cid[pid] += 1
        else:
            self.cid[pid] = 0
        letter = string.ascii_lowercase[self.cid[pid]]
        return pid + letter


class Record(object):

    def __init__(self, scan):
        self.scan = scan - 1
        self.shifts = pd.DataFrame()
        self.new_shifts = pd.DataFrame()
        self.correction_tally = {'case0': 0, 'case1': 0, 'case2': 0,
                                 'case3': 0, 'case4': 0, 'case5': 0}

    def count_case(self, case_num):
        self.correction_tally['case' + str(case_num)] += 1

    # Shift Correction Case Guide:
    # case0 - new object, local_shift and global_shift disagree, returns global
    # case1 - new object, returns local_shift
    # case2 - local disagrees with last head and global, returns last head
    # case3 - local disagrees with last head, returns local
    # case4 - local and last head agree, returns average of both
    # case5 - flow regions empty or at edge of frame, returns global_shift

    def record_shift(self, corr, gl_shift, l_heads, pc_shift, case):

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
        if len(self.new_shifts) > 0:
            self.new_shifts['uid'] = current_objects['uid']
            self.shifts = self.shifts.append(self.new_shifts)
            self.new_shifts = pd.DataFrame()

    def increment_scan(self):
        self.scan += 1


class Cell_tracks(object):

    def __init__(self, grid_list, grid_size=500):
        self.params = {'DBZ_THRESH': DBZ_THRESH,
                       'MIN_SIZE': MIN_SIZE,
                       'SEARCH_MARGIN': SEARCH_MARGIN,
                       'FLOW_MARGIN': FLOW_MARGIN,
                       'MAX_FLOW_MAG': MAX_FLOW_MAG,
                       'MAX_DISPARITY': MAX_DISPARITY,
                       'MAX_SHIFT_DISP': MAX_SHIFT_DISP}
        self.grid_size = grid_size
        self.grids = grid_list
        self.tracks, self.record = get_tracks(grid_list)

    def print_params(self):
        for key, val in self.params.items():
            print(key + ':', val)

    def animate(self, outfile_name, arrows=False):
        def animate_frame(nframe):
            """ Animate a single frame of gridded reflectivity including
            uids. """
            plt.clf()
            print("Frame:", nframe)
            grid = self.grids[nframe]
            display = pyart.graph.GridMapDisplay(grid)
            ax = fig_grid.add_subplot(111)
#            display.plot_basemap(lat_lines=np.arange(30, 46, .2),
#                                 lon_lines=np.arange(-110, -75, .4))
            display.plot_grid('reflectivity', level=3, vmin=-8, vmax=64,
                              cmap=pyart.graph.cm.NWSRef)

            frame_tracks = self.tracks[self.tracks['scan'] == nframe]
            for ind, uid in enumerate(frame_tracks['uid']):
                x = (frame_tracks['x'].iloc[ind])*self.grid_size
                y = (frame_tracks['y'].iloc[ind])*self.grid_size
                ax.annotate(uid, (y, x), fontsize=20)
                if arrows and ((nframe, uid) in self.record.shifts.index):
                    shift = self.record.shifts.loc[nframe, uid]['corrected']
                    shift = shift * self.grid_size
                    ax.arrow(y, x, shift[1], shift[0],
                             head_width=3*self.grid_size,
                             head_length=6*self.grid_size)
            return
        fig_grid = plt.figure(figsize=(10, 8))
        anim_grid = animation.FuncAnimation(fig_grid, animate_frame,
                                            frames=(len(self.grids) - 1))
        anim_grid.save(outfile_name,
                       writer='imagemagick', fps=1)

# _____________________________________________________________________________
# _____________________________Tracks Function_________________________________


def get_tracks(grids):
    start_time = datetime.datetime.now()
    start_scan = 0
    end_scan = len(grids) - 1

    counter = Counter()
    record = Record(start_scan)
    current_objects = None

    newRain = True

    print('Total scans in this list:', end_scan - start_scan + 1)

    grid2 = grids[start_scan].fields['reflectivity']['data']
    raw2 = grid2[3, :, :]
    frame2 = get_filtered_frame(grid2, MIN_SIZE, DBZ_THRESH)
    tracks = pd.DataFrame()

    for scan in range(start_scan + 1, end_scan + 1):
        record.increment_scan()
        raw1 = raw2
        frame1 = frame2
        grid2 = grids[scan].fields['reflectivity']['data']
        raw2 = grid2[3, :, :]
        frame2 = get_filtered_frame(grid2, MIN_SIZE, DBZ_THRESH)

        if np.max(frame1) == 0:
            newRain = True
            print('newRain')
            if 'current_objects' is not None:
                current_objects = None
            continue

        global_shift = get_global_shift(raw1, raw2, MAX_FLOW_MAG)
        pairs, record = get_pairs(frame1,
                                  frame2,
                                  global_shift,
                                  current_objects,
                                  record)
        obj_props = get_objectProp(frame1)

        if newRain:
            current_objects, counter = init_uids(
                frame1,
                frame2,
                pairs,
                counter
            )
            newRain = False
        else:
            current_objects, counter = update_current_objects(
                frame1,
                frame2,
                pairs,
                current_objects,
                counter
            )
        record.add_uids(current_objects)
        tracks = write_tracks(tracks, scan, current_objects, obj_props)
        # scan loop end

    record.shifts.set_index(['scan', 'uid'], inplace=True)
    record.shifts.sort_index(inplace=True)
    time_elapsed = datetime.datetime.now() - start_time
    print('\n')
    print('closing files')
    print('time elapsed', np.round(time_elapsed.seconds/60), 'minutes')

    return tracks, record
