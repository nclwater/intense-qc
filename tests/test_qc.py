from intense import utils
from intense.qc import Qc
from intense import gauge as ex
import os
from unittest import TestCase


class TestQc(TestCase):

    def test_get_flags(self):
        self.sample_data = "tests/sample_data"
        self.test_output = "tests/test_output"

        if not os.path.exists(self.test_output):
            os.mkdir(self.test_output)

        # use_daily_neighbours = False
        # use_monthly_neighbours = False

        folders_to_check = []
        for file in os.listdir(self.sample_data):
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
            f = utils.open_file(file_folders, files_to_process, file, self.sample_data, self.test_output)
            qc = Qc(ex.read_intense(f, only_metadata=False),
                    hourly_n_names=hourly_n_names,
                    hourly_n_dates=hourly_n_dates,
                    hourly_n_coords=hourly_n_coords,
                    hourly_n_paths=hourly_n_paths,
                    hourly_n_tree=hourly_n_tree,
                    etccdi_data_folder='tests/etccdi_data'
                    )
            qc.get_flags()
            output_folder = os.path.join(self.test_output, folder[:-4])
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            # for global run
            qc.write(output_folder + "/Flags")
            
            # Compare output QC flags file with benchmark
            benchmark_file_path = os.path.join(self.sample_data, "DE_02483/Flags/DE_02483_QC.txt")
            test_output_path = os.path.join(output_folder, "Flags/DE_02483_QC.txt")
            with open(benchmark_file_path, "r") as benchmark_file:
                benchmark_output = benchmark_file.readlines()
            with open(test_output_path, "r") as test_file:
                test_output = test_file.readlines()
            for benchmark_line, test_line in zip(benchmark_output, test_output):
                self.assertEqual(test_line.replace("(", "[").replace(")", "]"), 
                        benchmark_line.replace("(", "[").replace(")", "]"))
            