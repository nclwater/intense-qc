import unittest
from intense import rulebase

root_folder = 'tests/sample_data'
summary_path = 'tests/test_output/Rulebase_Summary.csv'


class TestRuleBase(unittest.TestCase):
    def test_main(self):
        rulebase.main(root_folder, summary_path, num_processes=1)