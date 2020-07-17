# -*- coding: utf-8 -*-

from get_metrics import StatisticExtractor
from automate_s2p import apply_to_all_one_plane_recordings
import os

def overwrite_iscell(exp_path):
    path = os.path.join(exp_path,'suite2p','plane0')
    se = StatisticExtractor(path)
    se._overwrite_iscell()

if __name__=="__main__":
    apply_to_all_one_plane_recordings("E:\\", overwrite_iscell)