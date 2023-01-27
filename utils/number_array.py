from typing import List
import numpy as np


def interval_intersection(intervals: List[List[List[int]]]) -> List[List[int]]:
    """
    Find the intersection of multiple interrupted timeseries, each of which contains multiple segments represented by a
    pair of start & end timestamp.

    Args:
        intervals: a 3-level list;
            1st level: each element represents a timeseries;
            2nd level: each element is a segment in a timeseries;
            3rd level: 2 elements are start & end timestamps of a segment

    Returns:
        The intersection is also a timeseries with the same format as one element in the input list.
    """
    if len(intervals) == 0:
        raise ValueError("The input list doesn't have any element.")
    if len(intervals) == 1:
        return intervals[0]

    # index indicating current position for each interval list
    pos_indices = np.zeros(len(intervals), dtype=int)
    # lengths of all ranges
    all_len = np.array([len(interval) for interval in intervals])

    result = []
    while np.all(pos_indices < all_len):
        # the startpoint of the intersection
        lo = max([intervals[interval_idx][pos_idx][0] for interval_idx, pos_idx in enumerate(pos_indices)])
        # the endpoint of the intersection
        endpoints = [intervals[interval_idx][pos_idx][1] for interval_idx, pos_idx in enumerate(pos_indices)]
        hi = min(endpoints)

        # save result if there is an intersection among segments
        if lo < hi:
            result.append([lo, hi])

        # remove the interval with the smallest endpoint
        pos_indices[np.argmin(endpoints)] += 1

    return result
