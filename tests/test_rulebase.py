from intense import rulebase
from . import TestCase
import os


class TestRuleBase(TestCase):
    def test_main(self):
        rulebase.main(self.sample_data, os.path.join(self.test_output, 'Rulebase_Summary.csv'))
