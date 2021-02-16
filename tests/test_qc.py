from intense import utils
from intense.qc import Qc
from intense.qc import read_intense_qc
from intense import gauge as ex
import os
from unittest import TestCase


class TestQc(TestCase):
    @classmethod
    def setUp(cls) -> None:
        cls.sample_data_path = "tests/sample_data"
        cls.test_output_path = "tests/test_output"

        if not os.path.exists(cls.test_output_path):
            os.mkdir(cls.test_output_path)

    def test_get_flags(self):
        qc = Qc(ex.read_intense(os.path.join(self.sample_data_path, 'gauges/DE_02483.txt'), only_metadata=False),
                use_hourly_neighbours=False)
        qc.get_flags()

        # for global run
        qc.write(self.test_output_path + "/Flags")
        read_intense_qc(self.test_output_path + '/Flags/DE_02483_QC.txt')

    def test_get_flags_hourly(self):

        hourly_n_names, hourly_n_dates, hourly_n_coords, hourly_n_paths, hourly_n_tree = \
            utils.create_kdtree_hourly_data('tests/sample_data/statlex_hourly.csv')

        qc = Qc(ex.read_intense(os.path.join(self.sample_data_path, 'gauges/DE_02483.txt'), only_metadata=False),
                use_hourly_neighbours=True,
                hourly_n_names=hourly_n_names,
                hourly_n_dates=hourly_n_dates,
                hourly_n_coords=hourly_n_coords,
                hourly_n_paths=hourly_n_paths,
                hourly_n_tree=hourly_n_tree,
                etccdi_data_folder='tests/etccdi_data',
                )
        qc.get_flags()

        # for global run
        qc.write(self.test_output_path + "/Flags")

        # Compare output QC flags file with benchmark
        benchmark_file_path = os.path.join(self.sample_data_path, "Flags/DE_02483_QC.txt")
        test_output_path = os.path.join(self.test_output_path, "Flags/DE_02483_QC.txt")
        with open(benchmark_file_path, "r") as benchmark_file:
            benchmark_output = benchmark_file.readlines()
        with open(test_output_path, "r") as test_file:
            test_output = test_file.readlines()
        for benchmark_line, test_line in zip(benchmark_output, test_output):
            self.assertEqual(test_line.replace("(", "[").replace(")", "]"),
                             benchmark_line.replace("(", "[").replace(")", "]"))

    def test_get_flags_hourly_daily_monthly(self):

        hourly_n_names, hourly_n_dates, hourly_n_coords, hourly_n_paths, hourly_n_tree = \
            utils.create_kdtree_hourly_data('tests/sample_data/statlex_hourly.csv')

        daily_names, daily_dates, daily_coords, daily_tree = utils.create_kdtree_daily_data(
            'tests/sample_data/statlex_daily.dat')

        monthly_names, monthly_dates, monthly_coords, monthly_tree = utils.create_kdtree_monthly_data(
            'tests/sample_data/statlex_monthly.dat'
        )

        qc = Qc(ex.read_intense(os.path.join(self.sample_data_path, 'gauges/DE_02483.txt'), only_metadata=False),
                use_hourly_neighbours=True,
                hourly_n_names=hourly_n_names,
                hourly_n_dates=hourly_n_dates,
                hourly_n_coords=hourly_n_coords,
                hourly_n_paths=hourly_n_paths,
                hourly_n_tree=hourly_n_tree,
                use_monthly_neighbours=True,
                monthly_names=monthly_names,
                monthly_dates=monthly_dates,
                monthly_coords=monthly_coords,
                monthly_tree=monthly_tree,
                monthly_path='tests/gpcc_data',
                use_daily_neighbours=True,
                daily_names=daily_names,
                daily_dates=daily_dates,
                daily_coords=daily_coords,
                daily_tree=daily_tree,
                daily_path='tests/gpcc_data',
                etccdi_data_folder='tests/etccdi_data',
                )
        qc.get_flags()

        # for global run
        qc.write(self.test_output_path + "/Flags")

        # Compare output QC flags file with benchmark
        benchmark_file_path = os.path.join(self.sample_data_path, "Flags/DE_02483_QC_hourly_daily_monthly.txt")
        test_output_path = os.path.join(self.test_output_path, "Flags/DE_02483_QC.txt")
        with open(benchmark_file_path, "r") as benchmark_file:
            benchmark_output = benchmark_file.readlines()
        with open(test_output_path, "r") as test_file:
            test_output = test_file.readlines()
        for benchmark_line, test_line in zip(benchmark_output, test_output):
            self.assertEqual(test_line.replace("(", "[").replace(")", "]"),
                             benchmark_line.replace("(", "[").replace(")", "]"))
