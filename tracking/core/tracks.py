"""
tint.tracks
===========
"""


import copy
import numpy as np
import pandas as pd
import datetime

from .grid_utils import get_grid_size, extract_grid_data
from .helpers import Record, Counter
from .phase_correlation import get_global_shift
from .matching import get_pairs
from .objects import init_current_objects, update_current_objects
from .objects import get_object_prop, write_tracks

# Parameter Defaults
FIELD_THRESH = 32
ISO_THRESH = 8
MIN_SIZE = 32
NEAR_THRESH = 4
SEARCH_MARGIN = 8
FLOW_MARGIN = 20
MAX_DISPARITY = 999
MAX_FLOW_MAG = 50
MAX_SHIFT_DISP = 15
GS_ALT = 1500


class Cell_tracks(object):
    """This is the main class in the module. It allows tracks
    objects to be built using lists of pyart grid objects."""

    def __init__(self, field='reflectivity'):
        self.params = {'FIELD_THRESH': FIELD_THRESH,
                       'MIN_SIZE': MIN_SIZE,
                       'SEARCH_MARGIN': SEARCH_MARGIN,
                       'FLOW_MARGIN': FLOW_MARGIN,
                       'MAX_FLOW_MAG': MAX_FLOW_MAG,
                       'MAX_DISPARITY': MAX_DISPARITY,
                       'MAX_SHIFT_DISP': MAX_SHIFT_DISP,
                       'NEAR_THRESH': NEAR_THRESH,
                       'ISO_THRESH': ISO_THRESH,
                       'GS_ALT': GS_ALT}

        self.field = field
        self.grid_size = None
        self.last_grid = None
        self.counter = None
        self.record = None
        self.current_objects = None
        self.tracks = pd.DataFrame()

        self.__saved_record = None
        self.__saved_counter = None
        self.__saved_objects = None

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

        if self.record is None:
            # tracks object being initialized
            grid_obj2 = next(grids)
            self.grid_size = get_grid_size(grid_obj2)
            self.counter = Counter()
            self.record = Record(grid_obj2)
        else:
            # tracks object being updated
            grid_obj2 = self.last_grid
            self.tracks.drop(self.record.scan + 1)  # last scan is overwritten

        if self.current_objects is None:
            newRain = True
        else:
            newRain = False

        raw2, frame2 = extract_grid_data(grid_obj2, self.field, self.grid_size,
                                         self.params)

        while grid_obj2 is not None:
            grid_obj1 = grid_obj2
            raw1 = raw2
            frame1 = frame2

            try:
                grid_obj2 = next(grids)
            except StopIteration:
                grid_obj2 = None

            if grid_obj2 is not None:
                self.record.update_scan_and_time(grid_obj1, grid_obj2)
                raw2, frame2 = extract_grid_data(grid_obj2,
                                                 self.field,
                                                 self.grid_size,
                                                 self.params)
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

            global_shift = get_global_shift(raw1, raw2, self.params)
            pairs = get_pairs(frame1,
                              frame2,
                              global_shift,
                              self.current_objects,
                              self.record,
                              self.params)

            if newRain:
                # first nonempty scan after a period of empty scans
                self.current_objects, self.counter = init_current_objects(
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

            obj_props = get_object_prop(frame1, grid_obj1, self.field,
                                        self.record, self.params)
            self.record.add_uids(self.current_objects)
            self.tracks = write_tracks(self.tracks, self.record,
                                       self.current_objects, obj_props)
#            gc.collect()
            del grid_obj1, raw1, frame1, global_shift, pairs, obj_props
            # scan loop end
        self.__load()
        time_elapsed = datetime.datetime.now() - start_time
        print('\n')
        print('time elapsed', np.round(time_elapsed.seconds/60, 1), 'minutes')
        return
