import unittest
import cProfile
from pstats import Stats


class TestCase(unittest.TestCase):
    def setUp(self):
        self.profile = cProfile.Profile()
        self.profile.enable()

    def tearDown(self):
        p = Stats(self.profile)
        p.strip_dirs()
        p.sort_stats('tottime')
        p.print_stats()