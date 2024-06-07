from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.max_disappeared = max_disappeared
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.objects_bb = OrderedDict()
        self.disappeared = OrderedDict()

    def register(self, bbox):
        new_id = self.next_object_id
        self.objects[self.next_object_id] = self._get_center(bbox)
        self.objects_bb[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

        return new_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.objects_bb[object_id]
        del self.disappeared[object_id]

    def _get_center(self, bbox):
        top, right, bottom, left = bbox
        cX = int((left + right) / 2.0)
        cY = int((top + bottom) / 2.0)
        return (cX, cY)

    def update(self, bbox_data):
        if len(bbox_data) == 0:
            # If no objects are detected, increase the disappeared count for all objects.
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return list(self.objects_bb.items())

        input_centroids = np.array([self._get_center(bbox) for bbox in bbox_data])

        if len(self.objects) != 0:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects_bb[object_id] = bbox_data[col]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        return list(self.objects_bb.items())

    def get_bounding_box(self, object_id):
        return self.objects_bb.get(object_id, None)
