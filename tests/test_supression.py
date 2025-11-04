import unittest

import numpy as np

from supression import (
    adaptive_cluster_diou_nms,
    bws,
    cluster_ait,
    cluster_diou_bws,
    cluster_diou_nms,
    nms,
)


class SuppressionModuleTests(unittest.TestCase):
    def test_nms_overlapping_boxes(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 9.0, 9.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)

        kept_boxes, kept_scores = nms(boxes, scores, iou_thresh=0.5)

        self.assertEqual(len(kept_boxes), 1)
        self.assertTrue(np.allclose(kept_boxes[0], boxes[0]))
        self.assertAlmostEqual(float(kept_scores[0]), 0.9)
        self.assertEqual(kept_boxes.dtype, np.float32)
        self.assertEqual(kept_scores.dtype, np.float32)

    def test_bws_merges_nearby_boxes(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.6, 0.4], dtype=np.float32)

        kept_boxes, kept_scores = bws(boxes, scores, iou_thresh=0.4)

        self.assertEqual(len(kept_boxes), 1)
        expected = np.array([0.4, 0.4, 10.4, 10.4], dtype=np.float32)
        self.assertTrue(np.allclose(kept_boxes[0], expected, atol=1e-3))
        self.assertAlmostEqual(float(kept_scores[0]), 0.6)

    def test_cluster_diou_nms_retains_distant(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.85, 0.8], dtype=np.float32)

        kept_boxes, kept_scores = cluster_diou_nms(boxes, scores, diou_thresh=0.5)

        self.assertEqual(len(kept_boxes), 3)
        self.assertTrue(np.allclose(kept_scores, np.array([0.9, 0.85, 0.8], dtype=np.float32)))

    def test_adaptive_cluster_diou_preserves_dense_group(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.5, 0.5, 10.5, 10.5],
                [20.0, 20.0, 30.0, 30.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.95, 0.9, 0.6], dtype=np.float32)

        kept_boxes, kept_scores = adaptive_cluster_diou_nms(
            boxes,
            scores,
            T0=0.45,
            alpha=0.8,
            k=2,
            score_ratio_thresh=0.8,
            diou_dup_thresh=0.5,
        )

        self.assertEqual(len(kept_boxes), 3)
        self.assertTrue(
            np.allclose(kept_scores, np.array([0.95, 0.9, 0.6], dtype=np.float32), atol=1e-4)
        )

    def test_cluster_ait_mixed_density(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.6, 0.6, 10.6, 10.6],
                [30.0, 30.0, 40.0, 40.0],
                [60.0, 60.0, 70.0, 70.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.96, 0.93, 0.7, 0.65], dtype=np.float32)

        kept_boxes, kept_scores = cluster_ait(
            boxes,
            scores,
            T0=0.5,
            alpha=0.2,
            k=3,
            lambda_weight=0.6,
        )

        self.assertEqual(len(kept_boxes), 3)
        self.assertTrue(np.allclose(kept_boxes[0], boxes[0]))
        self.assertTrue(
            np.allclose(kept_scores, np.array([0.96, 0.7, 0.65], dtype=np.float32), atol=1e-4)
        )

    def test_cluster_diou_bws_merges_high_affinity(self) -> None:
        boxes = np.array(
            [
                [0.0, 0.0, 12.0, 12.0],
                [1.0, 1.0, 13.0, 13.0],
                [2.0, 2.0, 14.0, 14.0],
                [3.0, 3.0, 15.0, 15.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.85, 0.8, 0.75], dtype=np.float32)

        kept_boxes, kept_scores = cluster_diou_bws(
            boxes,
            scores,
            affinity_thresh=0.5,
            lambda_weight=0.6,
        )

        self.assertEqual(len(kept_boxes), 1)
        self.assertAlmostEqual(float(kept_scores[0]), 0.9)

    def test_empty_inputs_return_empty_arrays(self) -> None:
        empty_boxes = []
        empty_scores = []

        for fn in (
            nms,
            bws,
            cluster_diou_nms,
            adaptive_cluster_diou_nms,
            cluster_ait,
            cluster_diou_bws,
        ):
            kept_boxes, kept_scores = fn(empty_boxes, empty_scores)
            self.assertEqual(kept_boxes.shape, (0, 4))
            self.assertEqual(kept_scores.shape, (0,))
            self.assertEqual(kept_boxes.dtype, np.float32)
            self.assertEqual(kept_scores.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
