import datetime
import numpy as np 
import os
import pandas
from netCDF4 import Dataset
from scipy import ndimage, optimize


#def plot_objects_label(labeled_image, xvalues, yvalues):

def get_filteredFrame(ncfile, scan_num, min_size):
    echo_height = get_convHeight(ncfile, scan_num)
    labeled_echo = ndimage.label(echo_height)[0]
    frame = clear_smallEchoes(labeled_echo, min_size)
    return frame
    

def get_classFrame(ncfile, scan_num):
    echo_height = get_convHeight(ncfile, scan_num)
    return get_vertical_class(echo_height)
    
    
def get_convHeight(ncfile, scan):
    """ Reads height and classification of data and replaces non-convective
    and missing pixels with zero. """
    dbz_height = ncfile.variables['zero_dbz_cont'][scan, :, :]
    steiner = ncfile.variables['steiner_class'][scan, :, :]
    dbz_height[(~steiner.mask) & (steiner.data != 2)] = 0
    dbz_height[dbz_height.mask] = 0
    return dbz_height
    

def clear_smallEchoes(label_image, min_size):
    flat_image = pandas.Series(label_image.flatten())
    flat_image = flat_image[flat_image > 0]
    size_table = flat_image.value_counts(sort = False)
    onePix_objects = size_table.keys()[size_table < min_size]
    
    for obj in onePix_objects:
        label_image[label_image == obj] = 0

    label_image = ndimage.label(label_image)
    return label_image[0]
    
    
def get_vertical_class(conv_height):
    min_level = (5, 15, 31)
    max_level = (14, 20, 40)
    
    for i, val in enumerate(min_level):
        conv_height[(conv_height >= min_level[i]) &
                    (conv_height <= max_level[i])] = i
                    
    return conv_height
    
    
#def change_baseEpoch(time_seconds, From_epoch, To_epoch=):
    
    

def get_matchPairs(image1, image2):
    nObjects1 = np.max(image1)
    nObjects2 = np.max(image2)
    
    if nObjects1 == 0:
        print('No echoes found in the first scan.')
        return
    elif nObjects2 == 0:
        zero_pairs = np.zeros(nObjects1)
        return zero_pairs
        
    obj_match = locate_allObjects(image1, image2)
    pairs = match_pairs(obj_match)
    return pairs
    
    
def match_pairs(obj_match):
    pairs = optimize.linear_sum_assignment(obj_match)
    for id1 in pairs[0]:
        if obj_match[id1, pairs[1][id1]] > max_disparity:
            pairs[1][id1] = -1
    pairs = pairs[1] + 1
    return pairs
    

def locate_allObjects(image1, image2):
    nObjects1 = np.max(image1)
    nObjects2 = np.max(image2)
    
    if (nObjects2 == 0) or (nObjects1 == 0):
        print('No echoes to track!')
        return
    
    obj_match = np.full((nObjects1, np.max((nObjects1, nObjects2))),
                        large_num, dtype = 'f')
    
    for obj_id1 in np.arange(nObjects1) + 1:
        obj1_extent = get_objExtent(image1, obj_id1)
        shift = get_std_flowVector(obj1_extent, image1, image2,
                                   flow_margin, stdFlow_mag)
        print(obj_id1, obj1_extent['obj_center'])
        print(shift)
        
        if 'current_objects' in globals():
            shift = correct_shift(shift, current_objects, obj_id1)
            
        search_box = predict_searchExtent(obj1_extent, shift, search_margin)
        search_box = check_searchBox(search_box, image2.shape)
        obj_found = find_objects(search_box, image2)
        disparity = get_disparity_all(obj_found, image2,
                                      search_box, obj1_extent)
        obj_match = save_objMatch(obj_id1, obj_found, disparity, obj_match)
        
    return obj_match
    

def correct_shift(this_shift, current_objects, object_id1):
    obj_index = current_objects['id2'] == object_id1
    last_heads = np.ma.append(current_objects['xhead'][obj_index],
                              current_objects['yhead'][obj_index])
    
    if (np.any(last_heads.mask) or
        all(np.logical_and(last_heads <= 1, last_heads >= -1))):
        return this_shift
    elif any(abs(this_shift-last_heads) > 4):
        return last_heads
    else:
        return (this_shift+last_heads)/2
    
    
def get_objExtent(labeled_image, obj_label):
    obj_index = np.argwhere(labeled_image == obj_label)
    
    xlength = np.max(obj_index[:, 0]) - np.min(obj_index[:, 0]) + 1
    ylength = np.max(obj_index[:, 1]) - np.min(obj_index[:, 1]) + 1
    
    obj_radius = np.max((xlength, ylength))/2
    obj_center = (np.min(obj_index[:, 0]) + obj_radius,
                  np.min(obj_index[:, 1]) + obj_radius)
    obj_area = len(obj_index[:, 0])
    
    obj_extent = {'obj_center':obj_center, 'obj_radius':obj_radius,
                  'obj_area':obj_area, 'obj_index':obj_index}
    
    return obj_extent
    
    
def get_objClass_extent(label_image, class_image, obj_label):
    objExtent = get_objExtent(label_image, obj_label)
    objClass = get_object_vertProfile(label_image, class_image, obj_label)
    objExtent['Cu_cong'] = objClass['Cu_cong']
    objExtent['Cu_deep'] = objClass['Cu_deep']
    objExtent['Cu_over'] = objClass['Cu_over']

    return objExtent
    
def get_objAmbientFlow(obj_extent, img1, img2, margin):
    r1 = obj_extent['obj_center'][0] - obj_extent['obj_radius'] - margin
    r2 = obj_extent['obj_center'][0] + obj_extent['obj_radius'] + margin
    c1 = obj_extent['obj_center'][1] - obj_extent['obj_radius'] - margin
    c2 = obj_extent['obj_center'][1] + obj_extent['obj_radius'] + margin
    r1 = np.int(r1)
    r2 = np.int(r2)
    c1 = np.int(c1)
    c2 = np.int(c2)

    dims = img1.shape
    if (r1 <= 0) or (c1 <= 0) or (r2 > dims[0]) or (c2 > dims[1]):
        return np.array((0, 0))
    
    flow_region1 = img1[r1:r2+1, c1:c2+1]
    flow_region2 = img2[r1:r2+1, c1:c2+1]

    return fft_flowVectors(flow_region1, flow_region2)
    
    
def get_std_flowVector(obj_extent, img1, img2, margin, magnitude):
    shift = get_objAmbientFlow(obj_extent, img1, img2, margin)
    shift[shift > magnitude] = magnitude
    shift[shift < -magnitude] = -magnitude
    return shift
    
    
def fft_flowVectors(im1, im2):
    if (np.max(im1) == 0) or (np.max(im2) == 0):
        return np.array((0, 0))
        
    crossCov = fft_crossCov(im1, im2)
#    if np.max(crossCov) == 0:
#        return np.array([1.2345, 1.2345])
    sigma = (1/8) * min(crossCov.shape)
    cov_smooth = ndimage.filters.gaussian_filter(crossCov, sigma)
    dims = im1.shape
    pshift = np.argwhere(cov_smooth == np.max(cov_smooth))[0]
    pshift = (pshift+1) - (dims[0]/2)
    return pshift
        
    
def fft_crossCov(im1, im2):
    fft1_conj = np.conj(np.fft.fft2(im1))
    fft2 = np.fft.fft2(im2)
    
#    if np.any(fft1_conj == 0) or np.any(fft2 == 0):
#        return np.zeros_like(fft2)
    normalize = abs(fft2*fft1_conj) #prevent divide by zero error
    normalize[normalize == 0] = 1
    C = (fft2*fft1_conj)/normalize
    crossCov = np.fft.ifft2(C)
    crossCov = np.real(crossCov)
    return fft_shift(crossCov)
    
def fft_shift(fft_mat):
    if type(fft_mat) is np.ndarray:
        rd2 = np.int(fft_mat.shape[0]/2)
        cd2 = np.int(fft_mat.shape[1]/2) 
        q1 = fft_mat[:rd2, :cd2]
        q2 = fft_mat[:rd2, cd2:]
        q3 = fft_mat[rd2:, cd2:]
        q4 = fft_mat[rd2:, :cd2]
        centered_t = np.concatenate((q4, q1), axis=0)
        centered_b = np.concatenate((q3, q2), axis=0)
        centered = np.concatenate((centered_b, centered_t), axis=1) 
        return centered
    else:
        print('input to fft_shift() should be a matrix')
        return

        
def predict_searchExtent(obj1_extent, shift, search_radius):
    shifted_center = obj1_extent['obj_center'] + shift
    
    x1 = shifted_center[0] - search_radius
    x2 = shifted_center[0] + search_radius + 1
    y1 = shifted_center[1] - search_radius
    y2 = shifted_center[1] + search_radius + 1
    x1 = np.int(x1)
    x2 = np.int(x2)
    y1 = np.int(y1)
    y2 = np.int(y2)

    return {'x1':x1, 'x2':x2, 'y1':y1, 'y2':y2,
            'center_pred':shifted_center, 'valid':True}


def check_searchBox(search_box, img_dims):
    if search_box['x1'] < 0:
        search_box['x1'] = 0

    if search_box['y1'] < 0:
        search_box['y1'] = 0

    if search_box['x2'] > img_dims[0]:
        search_box['x2'] = img_dims[0]

    if search_box['y2'] > img_dims[1]:
        search_box['y2'] = img_dims[1]

    if ((search_box['x2'] - search_box['x1'] < 5) 
        or (search_box['y2'] - search_box['y1'] < 5)):
        search_box['valid'] = False
 
    return search_box
        
        
def find_objects(search_box, image2):
    if not search_box['valid']:
        obj_found = np.array(-1)
    else:
        search_area = image2[search_box['x1']:search_box['x2'],
                             search_box['y1']:search_box['y2']]
        obj_found = np.unique(search_area)

    return obj_found
    
    
def get_disparity_all(obj_found, image2, search_box, obj1_extent):
    if np.max(obj_found) <= 0:
        disparity = np.array([large_num])
    else:
        obj_found = obj_found[obj_found > 0]
        if len(obj_found) == 1:
            disparity = get_disparity(obj_found, image2,
                                      search_box, obj1_extent)
            if(disparity <= 3):
                disparity = np.array([0])
        else:
            disparity = get_disparity(obj_found, image2,
                                      search_box, obj1_extent)
            
    return disparity
    

def save_objMatch(obj_id1, obj_found, disparity, obj_match):
    if np.max(obj_found) > 0 and np.all(disparity < max_disparity):
        obj_found = obj_found[obj_found > 0]
        obj_found = obj_found - 1
        obj_id1 = obj_id1 - 1
        obj_match[obj_id1, obj_found] = disparity
    return obj_match
    

def get_disparity(obj_found, image2, search_box, obj1_extent):
    dist_pred = np.empty(0)
    dist_actual = np.empty(0)
    change = np.empty(0)
    for target_obj in obj_found:
        target_extent = get_objExtent(image2, target_obj)
        
        euc_dist = euclidean_dist(target_extent['obj_center'],
                                  search_box['center_pred'])
        dist_pred = np.append(dist_pred, euc_dist)
        
        euc_dist = euclidean_dist(target_extent['obj_center'],
                                  obj1_extent['obj_center'])
        dist_actual = np.append(dist_actual, euc_dist)
        
        size_changed = get_sizeChange(target_extent['obj_area'],
                                      obj1_extent['obj_area'])
        change = np.append(change, size_changed)
    
    disparity = dist_pred + change
    return disparity


def euclidean_dist(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dist = np.sqrt(sum((vec1-vec2)**2))
    return dist
    
def get_sizeChange(x, y):
    if (x < 5) and (y < 5):
        return(0)
    elif x >= y:
        return (x/y -1)
    else:
        return(y/x -1)


def create_outNC(ofile, max_obs):
    if os.path.isfile('ofile'):
        print('removing existing file', os.path.basename(ofile))
        os.remove(ofile)
        
    deflat = 9
    outNC = Dataset(ofile, 'w')
     
    outNC.createDimension('echo_id', None)
    #dim_echo.longname = 'unique id of convection echo'
    outNC.createDimension('records', max_obs)
    #dim_obs.longname = 'observation records'
    outNC.createDimension('time', None)
    #dim_time.units = 'seconds since 1970-01-01 00:00:00 UTC'
    #dim_time.longname = 'time of the scan'
    outNC.createDimension('stat', 4)
    #dim_stat.longname = 'object survival vector; lived, died, born, total'
    
    outNC.createVariable(
        'survival', datatype='i', dimensions=('stat', 'time'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'duration', datatype='i', dimensions=('echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'origin', datatype='i',
        dimensions='echo_id', fill_value=0
        )
    
    outNC.createVariable(
        'merged', datatype='i',
        dimensions='echo_id', fill_value=0
        )
    
    outNC.createVariable(
        'record_time', datatype='i', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'x_dist', datatype='f', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'y_dist', datatype='f', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'x', datatype='i', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'y', datatype='i', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'area', datatype='i', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'Cu_cong', datatype='i', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'Cu_deep', datatype='i', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
    outNC.createVariable(
        'Cu_over', datatype='i', dimensions=('records', 'echo_id'),
        fill_value=-999, complevel=deflat
        )
        
    #for CF standards
    #dim_echo.cf_role =  'trajectory_id'
    outNC.setncattr('featureType', 'trajectory')
    
    description = """The CPOL radar echoes of convective types were separated
                     using Steiner classification scheme and tracked. Merging
                     and splitting is added with echo ids."""
    outNC.setncattr('_description', description)
    outNC.setncattr('_creator', 'Bhupendra Raut')
    outNC.setncattr('_url', 'www.baraut.info')
    outNC.setncattr('_date_created', str(datetime.date.today()))
    
    return outNC
    

def write_update(outNC, current_objects, obj_props, obs_time):
    nobj = len(current_objects['id1'])
    
    for object in range(nobj):
        nc_start = (current_objects['obs_num'][object],
                    current_objects['uid'][object])
        
        outNC.variables['record_time'][nc_start] = obs_time
        outNC.variables['x'][nc_start] = obj_props['x'][object]
        outNC.variables['y'][nc_start] = obj_props['y'][object]
        outNC.variables['x_dist'][nc_start] = obj_props['xdist'][object]
        outNC.variables['y_dist'][nc_start] = obj_props['ydist'][object]
        outNC.variables['area'][nc_start] = obj_props['area'][object]
        outNC.variables['Cu_cong'][nc_start] = obj_props['Cu_cong'][object]
        outNC.variables['Cu_deep'][nc_start] = obj_props['Cu_deep'][object]
        outNC.variables['Cu_over'][nc_start] = obj_props['Cu_over'][object]

    write_duration(outNC, current_objects)
    

def write_duration(outNC, current_objects):
    nobj = len(current_objects['id1'])
    
    for obj in range(nobj):
        if current_objects['id2'][obj] == 0:
            uid = current_objects['uid'][obj]
            outNC.variables['duration'][uid] = current_objects['obs_num'][obj]
            outNC.variables['origin'][uid] = current_objects['origin'][obj]
            merged_in = check_merging(obj, current_objects, obj_props)
            outNC.variables['merged'][uid] = merged_in


def check_merging(dead_obj_id1, current_objects, obj_props):
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


def write_survival(outNC, survival_stat, time, scan):
    outNC.variables['survival'][:5, scan] = survival_stat
    outNC.variables['record_time'][scan] = time


def init_uids(first_frame, second_frame, pairs):
    nobj = np.max(first_frame)
    
    id1 = np.arange(nobj) + 1
    uid = next_uid(count=nobj)
    id2 = pairs
    obs_num = np.ones(nobj, dtype = 'i')
    origin = np.zeros(nobj, dtype = 'i')
    
    current_objects = {'id1':id1, 'uid':uid, 'id2':id2,
                       'obs_num':obs_num, 'origin':origin}
    current_objects = attach_xyheads(first_frame, second_frame,
                                     current_objects)
    return current_objects


def attach_xyheads(frame1, frame2, current_objects):
    
    nobj = len(current_objects['uid'])
    xhead = np.ma.empty(0)
    yhead = np.ma.empty(0)
    NA = np.ma.array([-999], mask = [True])
    
    for obj in range(nobj):
        if ((current_objects['id1'][obj] > 0)
            and (current_objects['id2'][obj] > 0)):
            
            center1 = get_objectCenter(current_objects['id1'][obj], frame1)
            center2 = get_objectCenter(current_objects['id1'][obj], frame2)
            xhead = np.ma.append(xhead, center1[0] - center2[0])
            yhead = np.ma.append(yhead, center1[1] - center2[1])
        else:
            xhead = np.ma.append(xhead, NA)
            yhead = np.ma.append(yhead, NA)
            
    current_objects['xhead'] = xhead
    current_objects['yhead'] = yhead
    return current_objects
                  
    
def get_objectCenter(obj_id, labeled_image):
    obj_index = np.argwhere(labeled_image == obj_id)
    center_x = np.int(np.median(obj_index[:, 0]))
    center_y = np.int(np.median(obj_index[:, 1]))
    return np.array((center_x, center_y))
    

def update_current_objects(frame1, frame2, pairs, old_objects):
    nobj = np.max(frame1)
    id1 = np.arange(nobj) + 1
    uid = np.array([], dtype = 'i')
    obs_num = np.array([], dtype = 'i')
    origin = np.array([], dtype = 'i')
    
    for obj in np.arange(nobj) + 1:
        if obj in old_objects['id2']:
            obj_index = old_objects['id2'] == obj
            uid = np.append(uid, old_objects['uid'][obj_index])
            obs_num = np.append(obs_num, old_objects['obs_num'][obj_index] + 1)
            origin = np.append(origin, old_objects['origin'][obj_index])
        else:
            uid = np.append(uid, next_uid())
            obs_num = np.append(obs_num, 1)
            origin = np.append(origin,
                               get_origin_uid(obj, frame1, old_objects))
            
    id2 = pairs
    current_objects = {'id1':id1, 'uid':uid, 'id2':id2,
                       'obs_num':obs_num, 'origin':origin}
    current_objects = attach_xyheads(frame1, frame2, current_objects)
    return current_objects
    

def get_origin_uid(obj, frame1, old_objects):
    origin_id = find_origin(obj, frame1)
    if origin_id == 0:
        return 0
        
    origin_index = np.argwhere(old_objects['id2'] == origin_id)
    
    if not origin_id in old_objects['id2']:
        return 0
        
    origin_uid = old_objects['uid'][origin_index]
    return origin_uid
    

def find_origin(id1_newObj, frame1):
    if np.max(frame1) == 1:
        return 0
        
    object_ind = np.argwhere(frame1 == id1_newObj)
    object_size = object_ind.shape[0]
    
    neighbour_ind = np.argwhere((frame1 > 0) & (frame1 != id1_newObj))
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
    
    nearest_object_id = neighbour_id[neighbour_dist < 4]
   # the_nearest_object = neighbour_id[neighbour_dist == min(neighbour_dist)]
    
    if len(nearest_object_id) == 0:
        return 0
    
    neigh_objects = np.unique(nearest_object_id)
    for object in neigh_objects:
        nearest_object_size = len(frame1[frame1 == object])
        size_ratio = np.append(size_ratio, nearest_object_size/object_size)
        size_diff = np.append(size_diff, nearest_object_size - object_size)
        
    big_ratio_obj = neigh_objects[size_ratio == max(size_ratio)]
    big_diff_obj = neigh_objects[size_diff == max(size_diff)]
    
    if(big_ratio_obj == big_diff_obj):
        return big_diff_obj[0]
    else:
        return big_diff_obj[0]


def next_uid(count = 1):
    global uid_counter
    this_uid = uid_counter + np.arange(count) + 1
    uid_counter = uid_counter + count
    return this_uid
    

def get_objectProp(image1, class1, xyDist):
    id1 = np.empty(0, dtype = 'i')
    x = np.empty(0, dtype = 'i')
    y = np.empty(0, dtype = 'i')
    area = np.empty(0, dtype = 'i')
    cu_cong = np.empty(0, dtype = 'i')
    cu_deep = np.empty(0, dtype = 'i')
    cu_over = np.empty(0, dtype = 'i')
    
    nobj = np.max(image1)
    
    for obj in np.arange(nobj) + 1:
        obj_index = np.argwhere(image1 == obj)
        id1 = np.append(id1, obj)
        x = np.append(x, np.floor(np.median(obj_index[:, 0])))
        y = np.append(y, np.floor(np.median(obj_index[:, 1])))
        area = np.append(area, len(obj_index[:, 1]))
        
        obj_class = get_object_vertProfile(image1, class1, obj_label = obj)
        
        cu_cong = np.append(cu_cong, obj_class['Cu_cong'])
        cu_deep = np.append(cu_deep, obj_class['Cu_deep'])
        cu_over = np.append(cu_over, obj_class['Cu_over'])
        
    objprop = {'id1':id1, 'x':x, 'y':y, 'area':area,
               'Cu_cong':cu_cong, 'Cu_deep':cu_deep, 'Cu_over':cu_over}
    objprop = attach_xyDist(objprop, xyDist['x'], xyDist['y'])
    return objprop
    
    
def get_object_vertProfile(label_image, class_image, obj_label):
    obj_class = class_image[label_image == obj_label]
    nCu_cong = len(obj_class[obj_class == 1])
    nCu_deep = len(obj_class[obj_class == 2])
    nCu_over = len(obj_class[obj_class == 3])
    return {'Cu_cong':nCu_cong, 'Cu_deep':nCu_deep, 'Cu_over':nCu_over}


def attach_xyDist(obj_props, xdist, ydist):
    obj_props['xdist'] = xdist[obj_props['x'].astype(int)]
    obj_props['ydist'] = ydist[obj_props['y'].astype(int)]
    return obj_props
    

def survival_stats(pairs, num_obj2):
    obj_lived = len(pairs[pairs > 0])
    obj_died = len(pairs) - obj_lived
    obj_born = num_obj2 - obj_lived
    return {'lived':obj_lived, 'died':obj_died,
            'born':obj_born, 'total':num_obj2}
            

#______________________________________________________________________________
#________________________Settings for tracking method__________________________
start_time = datetime.datetime.now()

search_margin = 4
flow_margin = 4
stdFlow_mag = 5
min_signif_movement = 2
large_num = 1000
max_obs = 60
min_size = 2
max_disparity = 15
#______________________________________________________________________________

file_list = ['/home/mhpicel/R Scripts/cpol_2D_2004-11-03.nc']

for infile_name in file_list:
    outfile_name = '/home/mhpicel/Desktop/test_tracks_py.nc'
    print('Opening output file', outfile_name)
    
    outNC = create_outNC(outfile_name, max_obs)
    uid_counter = 0
    
    #read x, y and time from the file
    ncfile = Dataset(infile_name, 'r', format='NETCDF4')
    x = ncfile.variables['x'][:]
    y = ncfile.variables['y'][:]
    
    time = ncfile.variables['time'][:]
    #time = change_baseEpoch(time, From_epoch=)
    
    start_scan = 1
    #end_scan = len(time)
    end_scan = 5
    
    newRain = True
    
    print('Total scans in this file', end_scan - start_scan + 1)
    #pb
    
    frame2 = get_filteredFrame(ncfile, start_scan, min_size)
    class2 = get_classFrame(ncfile, start_scan)
    
    for scan in range(start_scan + 1, end_scan + 1):
        #set pb
        
        frame1 = frame2
        class1 = class2
        
        frame2 = get_filteredFrame(ncfile, scan, min_size)
        class2 = get_classFrame(ncfile, scan)
        
        if scan == end_scan:
            frame2 = np.zeros_like(frame2)
        
        if np.max(frame1) == 0:
            newRain = True
            if 'current_objects' in globals():
                del current_objects
                
            write_survival(outNC, survival_stat = np.zeros(4),
                           time=time[scan], scan=scan)
            continue
        
        pairs = get_matchPairs(frame1, frame2)
        obj_props = get_objectProp(frame1, class1, {'x':x, 'y':y})
        
        if newRain:
            current_objects = init_uids(frame1, frame2, pairs)
            
            num_obj1 = np.max(frame1)
            survival = [0, 0, num_obj1, num_obj1]
            
            if scan == start_scan + 1:
                survival = [-999, -999, -999, num_obj1]

            write_survival(outNC, survival_stat=survival,
                           time=time[scan-1], scan=scan-1)
        
            newRain = False
        
        else:
            current_objects = update_current_objects(frame1, frame2,
                                                     pairs, current_objects)
        
        write_update(outNC, current_objects, obj_props, time[scan-1])
    
        #Survival for frame2
        num_obj2 = np.max(frame2)
        obj_survival = survival_stats(pairs, num_obj2)
        write_survival(outNC, survival_stat=obj_survival,
                       time=time[scan], scan=scan)
        
        #scan loop end
        
    print('\n')
    print('closing files')
    ncfile.close()   
    #write unlimited dim and close
    dim_echo[0:uid_counter] = np.arange(uid_counter) + 1
    outNC.close()
    
    time_elapsed = datetime.datetime.now() - start_time
    print('time elapsed', np.round(time_elapsed.seconds/60), 'minutes')