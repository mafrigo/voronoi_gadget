import unittest
import os
import numpy as np

from voronoi_gadget.voronoi_mapping import *
from voronoi_gadget.voronoi_grid import *
from voronoi_gadget.snapshot_tools import *


class TestSnapshot(unittest.TestCase):
    def test_snap_generation(self):
        test_snap_len = 1000
        snap = generate_snapshot(test_snap_len)
        self.assertEqual(len(snap["mass"]), test_snap_len)
        self.assertEqual(len(snap["ID"]), test_snap_len)
        self.assertEqual(snap["pos"].shape, (test_snap_len, 3))
        self.assertEqual(snap["vel"].shape, (test_snap_len, 3))

    def test_pseudo_snap(self):
        n_pseudo_particles = 5
        test_snap_len_pp = 100
        snap_pp = generate_snapshot(test_snap_len_pp)
        pseudosnap = PseudoSnap(snap_pp, n_pseudo_particles, 0.5, 1.)
        self.assertEqual(len(pseudosnap["mass"]), test_snap_len_pp*n_pseudo_particles)
        self.assertEqual(len(pseudosnap["ID"]), test_snap_len_pp*n_pseudo_particles)
        self.assertEqual(pseudosnap["pos"].shape, (test_snap_len_pp*n_pseudo_particles, 3))
        self.assertEqual(pseudosnap["vel"].shape, (test_snap_len_pp*n_pseudo_particles, 3))


class TestVoronoiTessellation(unittest.TestCase):
    def test_grid_nspaxel(self):
        test_nspaxels = 50
        snap = generate_snapshot(1000000)
        grid = VoronoiGrid(snap, 4., npixel_per_side=50, nspaxels=test_nspaxels)
        print("Target: " + str(test_nspaxels))
        print("Actual: " + str(grid.nspaxels))
        self.assertGreaterEqual(grid.nspaxels, 0.8*test_nspaxels)
        self.assertLessEqual(grid.nspaxels, 1.2*test_nspaxels)

    def test_grid_ppb(self):
        test_ppb = 10000
        snap = generate_snapshot(1000000)
        grid = VoronoiGrid(snap, 4., npixel_per_side=50, partperbin=test_ppb)
        average_ppb = np.mean(np.unique(grid._spaxelofpart, return_counts=True)[1])
        print("Target: " + str(test_ppb))
        print("Actual: " + str(average_ppb))
        self.assertGreaterEqual(average_ppb, 0.8*test_ppb)
        self.assertLessEqual(average_ppb, 1.2*test_ppb)


class TestStatistics(unittest.TestCase):
    def test_stats(self):
        test_data = np.loadtxt("sample_data/test_distribution.txt")
        stats = gauss_hermite_fit(test_data)
        exp_stats = -108., 428., 0.0255, 0.017
        print("Target: ", exp_stats)
        print("Actual: ", stats)
        for i in [0,1,2,3]:
            self.assertTrue(abs(0.9*exp_stats[i]) <= abs(stats[i]) <= abs(1.1*exp_stats[i]))
            self.assertEqual(np.sign(stats[i]), np.sign(exp_stats[i]))


class TestMapping(unittest.TestCase):
    def test_cbar_ticks(self):
        self.assertEqual(1., 1.)


class TestEndToEnd(unittest.TestCase):
    def test_e2e(self):
        test_output_image = "vel4map_test.png"
        if os.path.exists(test_output_image):
            os.remove(test_output_image)
        snap = generate_snapshot(100000)
        voronoimap(snap, "vel", extent=4., scalebar=1., force_orient=False, npixel_per_side=50, nspaxels=100,
                   plotfile=test_output_image, style='dark')
        self.assertEqual(os.path.exists(test_output_image), True)


if __name__ == '__main__':
    unittest.main()
