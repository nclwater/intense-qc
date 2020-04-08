from intense import rulebase
from . import TestCase

root_folder = 'tests/test_output'
summary_path = 'tests/test_output/Rulebase_Summary.csv'


class TestRuleBase(TestCase):
    def test_main(self):
        rulebase.main(root_folder, summary_path)
