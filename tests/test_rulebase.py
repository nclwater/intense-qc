#import sys
#sys.path.append(".")

import unittest
from intense import rulebase

root_folder = 'tests/test_output'
summary_path = 'tests/test_output/Rulebase_Summary.csv'


class TestRuleBase(unittest.TestCase):
    def test_main(self):
        rulebase.main(root_folder, summary_path, 1)

if __name__ == '__main__':
    unittest.main()
