import os
from intense import rulebase
from unittest import TestCase

root_folder = 'tests/sample_data'
summary_path = 'tests/test_output/Rulebase_Summary.csv'


class TestRuleBase(TestCase):
    def test_apply_all(self):
        rulebase.apply_all([os.path.join(root_folder, 'flags/DE_02483_QC.txt')], summary_path)
        
        # Compare output QC'd data file with benchmark
        benchmark_file_path = os.path.join(root_folder, "QCd_Data/DE_02483.txt")
        test_output_path = "tests/test_output/QCd_Data/DE_02483.txt"
        with open(benchmark_file_path, "r") as benchmark_file:
            benchmark_output = benchmark_file.readlines()
        with open(test_output_path, "r") as test_file:
            test_output = test_file.readlines()
        self.assertEqual(test_output, benchmark_output)

    def test_apply(self):
        rulebase.apply_rulebase(os.path.join(root_folder, 'flags/DE_02483_QC.txt'), os.path.dirname(summary_path),
                                station_id_suffix='_QCd')
