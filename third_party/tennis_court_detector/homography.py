import cv2
import numpy as np
from scipy.spatial import distance

from .court_reference import CourtReference

court_ref = CourtReference()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

court_conf_ind = {}
for i in range(len(court_ref.court_conf)):
    conf = court_ref.court_conf[i + 1]
    inds = [court_ref.key_points.index(conf[j]) for j in range(4)]
    court_conf_ind[i + 1] = inds


def get_trans_matrix(points):
    """Repair keypoints using best 4-point homography configuration."""
    matrix_trans = None
    dist_max = np.inf
    for conf_ind in range(1, 13):
        conf = court_ref.court_conf[conf_ind]
        inds = court_conf_ind[conf_ind]
        inters = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
        if any(x is None or x[0] is None for x in inters):
            continue
        matrix, _ = cv2.findHomography(np.float32(conf), np.float32(inters), method=0)
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
        dists = []
        for i in range(14):
            if i not in inds and points[i] is not None and points[i][0] is not None:
                dists.append(float(distance.euclidean(points[i], trans_kps[i].reshape(-1))))
        if not dists:
            continue
        dist_median = float(np.mean(dists))
        if dist_median < dist_max:
            matrix_trans = matrix
            dist_max = dist_median
    return matrix_trans
