import os
from intense import rulebase
from unittest import TestCase

root_folder = 'tests/sample_data'
summary_path = 'tests/test_output/Rulebase_Summary.csv'


class TestRuleBase(TestCase):
    def test_main(self):
        rulebase.find_and_apply_rulebase(root_folder, summary_path)
        
        # Compare output QC'd data file with benchmark
        benchmark_file_path = os.path.join(root_folder, "DE_02483/QCd_Data/DE_02483.txt")
        test_output_path = "tests/test_output/QCd_Data/DE_02483.txt"
        with open(benchmark_file_path, "r") as benchmark_file:
            benchmark_output = benchmark_file.readlines()
        with open(test_output_path, "r") as test_file:
            test_output = test_file.readlines()
        assert test_output == benchmark_output
        
