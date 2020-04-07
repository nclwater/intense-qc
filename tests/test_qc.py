import unittest

import sys
sys.path.append(".")

from intense import utils
from intense.qc import Qc
from intense import intense as ex
import os

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class TestQc(unittest.TestCase):
    def test_get_flags(self):

        orig_folder = "tests/sample_data"
        qc_folder = "tests/test_output"
        if not os.path.exists(qc_folder):
            os.mkdir(qc_folder)

        # use_daily_neighbours = False
        # use_monthly_neighbours = False

        folders_to_check = []
        for file in os.listdir(orig_folder):
            if file.endswith(".zip"):
                folders_to_check.append(file)

        hourly_n_names, hourly_n_dates, hourly_n_coords, hourly_n_paths, hourly_n_tree = \
            utils.create_kdtree_hourly_data('tests/sample_data/statlex_hourly.csv')

        # if use_daily_neighbours:
        #     daily_names, daily_dates, daily_coords, tree = utils.create_kdtree_daily_data()
        # if use_monthly_neighbours:
        #     monthly_names, monthly_dates, monthly_coords, monthly_tree = utils.create_kdtree_monthly_data()

        # Serial processing
        # files_to_process, file_folders = utils.find_files_to_process(folders_to_check, qc_folder, orig_folder)
        files_to_process = ["DE_02483.txt"]
        file_folders = ["DE_02483.zip"]
        for file, folder in zip(files_to_process, file_folders):
            f = utils.open_file(file_folders, files_to_process, file, orig_folder, qc_folder)
            qc = Qc(ex.read_intense(f, only_metadata=False, opened=True),
                    hourly_n_names=hourly_n_names,
                    hourly_n_dates=hourly_n_dates,
                    hourly_n_coords=hourly_n_coords,
                    hourly_n_paths=hourly_n_paths,
                    hourly_n_tree=hourly_n_tree,
                    etccdi_data_folder='tests/etccdi_data'
                    )
            qc.get_flags()

            # for global run
            qc.write(qc_folder + "/" + folder[:-4] + "/Flags")


if __name__ == '__main__':
    unittest.main()