from intense import rulebase
from unittest import TestCase

root_folder = 'tests/sample_data'
summary_path = 'tests/test_output/Rulebase_Summary.csv'


class TestRuleBase(TestCase):
    def test_main(self):
        rulebase.main(root_folder, summary_path)
