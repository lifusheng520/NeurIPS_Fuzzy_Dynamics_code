from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.evaluation import (
    automatic_semantic_event_scores,
    binary_ranking_metrics,
    reconstruction_metrics,
    rollout_metrics,
    semantic_alignment_metrics,
)
from src.evaluation.events import load_semantic_event_definitions
from src.fuzzy_dynamics import phi_a, prediction_refinement_score, route_score


class EvaluationMetricsTest(unittest.TestCase):
    def test_paper_semantic_score_definitions(self) -> None:
        attention = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])
        self.assertTrue(torch.equal(route_score(attention), torch.tensor([[5.0, 13.0]])))

        gamma_change = torch.tensor([[-0.2, 0.4, 0.9]])
        uncertainty_drop = torch.tensor([[0.5, -0.3, 0.4]])
        expected = torch.tensor([[0.0, 0.0, 0.6]])
        self.assertTrue(torch.allclose(phi_a(gamma_change, uncertainty_drop), expected))

        margin = torch.tensor([[[0.1], [0.0], [0.4], [0.9]]])
        uncertainty = torch.tensor([[[0.8], [0.3], [0.5], [0.1]]])
        self.assertTrue(
            torch.allclose(
                prediction_refinement_score(margin, uncertainty),
                torch.tensor([[0.0, 0.0, 0.4472136]]),
            )
        )

    def test_external_events_align_by_uid_not_row_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            records = [
                {
                    "uid": "a",
                    "events": {"hop_transition": [0, None, 1]},
                    "provenance": {
                        "source": "human-v1",
                        "independent_of_training": True,
                    },
                },
                {
                    "uid": "b",
                    "events": {"hop_transition": [1, 0, 0]},
                    "provenance": {
                        "source": "human-v1",
                        "independent_of_training": True,
                    },
                },
            ]
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")

            definitions, manifest = load_semantic_event_definitions(
                path, [{"uid": "b"}, {"uid": "a"}], num_layers=3
            )

            hop = definitions["hop_transition"]
            self.assertEqual(hop["labels"].tolist(), [[True, False, False], [False, False, True]])
            self.assertEqual(hop["valid_mask"].tolist(), [[True, True, True], [True, False, True]])
            self.assertTrue(hop["independent"])
            self.assertEqual(manifest["matched_queries"], 2)

    def test_vector_baseline_and_block_macro_r2(self) -> None:
        target = torch.tensor(
            [
                [
                    [0.0, 10.0, 20.0, 30.0, 40.0],
                    [1.0, 11.0, 21.0, 31.0, 41.0],
                    [2.0, 12.0, 22.0, 32.0, 42.0],
                    [3.0, 13.0, 23.0, 33.0, 43.0],
                ]
            ],
            dtype=torch.float64,
        )
        predicted = target + torch.tensor([0.5, 0.5, 1.0, 0.0, 0.5])
        dimensions = {"z": 2, "concept": 1, "belief": 1, "uncertainty": 1}

        result = reconstruction_metrics(predicted, target, dimensions)

        self.assertAlmostEqual(result["mse"], 0.35)
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["r2"], 0.72)
        self.assertAlmostEqual(result["components"]["z"]["r2"], 0.8)
        self.assertAlmostEqual(result["components"]["concept"]["r2"], 0.2)
        self.assertAlmostEqual(result["components"]["belief"]["r2"], 1.0)
        self.assertAlmostEqual(result["components"]["uncertainty"]["r2"], 0.8)
        self.assertAlmostEqual(result["macro_r2"], 0.7)
        self.assertIsNone(result["layer_macro_r2"])

    def test_constant_block_has_undefined_r2_without_nan(self) -> None:
        target = torch.tensor(
            [
                [
                    [0.0, 10.0, 20.0, 30.0, 40.0],
                    [1.0, 11.0, 20.0, 31.0, 41.0],
                    [2.0, 12.0, 20.0, 32.0, 42.0],
                    [3.0, 13.0, 20.0, 33.0, 43.0],
                ]
            ],
            dtype=torch.float64,
        )
        predicted = target + torch.tensor([0.5, 0.5, 1.0, 0.0, 0.5])
        dimensions = {"z": 2, "concept": 1, "belief": 1, "uncertainty": 1}

        result = reconstruction_metrics(predicted, target, dimensions)

        self.assertAlmostEqual(result["r2"], 0.65)
        self.assertIsNone(result["components"]["concept"]["r2"])
        self.assertFalse(result["components"]["concept"]["r2_defined"])
        self.assertIsNone(result["macro_r2"])
        self.assertAlmostEqual(result["macro_r2_available_components"], 13.0 / 15.0)
        json.dumps(result, allow_nan=False)

    def test_binary_ranking_metrics_are_tie_aware(self) -> None:
        result = binary_ranking_metrics(
            torch.tensor([0.1, 0.7, 0.4, 0.8]),
            torch.tensor([0, 1, 1, 0]),
        )
        self.assertAlmostEqual(result["auroc"], 0.5)
        self.assertAlmostEqual(result["average_precision"], 7.0 / 12.0)

        tied = binary_ranking_metrics(
            torch.full((4,), 0.5),
            torch.tensor([1, 1, 0, 0]),
        )
        self.assertAlmostEqual(tied["auroc"], 0.5)
        self.assertAlmostEqual(tied["average_precision"], 0.5)

    def test_binary_ranking_rejects_degenerate_or_invalid_labels(self) -> None:
        one_class = binary_ranking_metrics(
            torch.arange(4, dtype=torch.float32), torch.ones(4)
        )
        self.assertFalse(one_class["available"])
        self.assertIsNone(one_class["auroc"])
        self.assertIsNone(one_class["average_precision"])
        with self.assertRaises(ValueError):
            binary_ranking_metrics(torch.arange(2.0), torch.tensor([0.0, 0.2]))

    def test_automatic_semantic_alignment_for_documented_modes(self) -> None:
        bridge = torch.tensor([[[-4.0], [-3.0], [-2.0], [-3.0], [-4.0]]])
        answer = torch.tensor([[[-4.0], [-5.0], [-4.0], [-3.0], [-2.0]]])
        uncertainty = torch.tensor([[[0.8], [0.6], [0.4], [0.6], [0.4]]])
        margin = torch.tensor([[[0.4], [0.3], [0.5], [0.7], [0.9]]])
        memberships = torch.tensor(
            [
                [
                    [0.300, 0.300, 0.300, 0.050, 0.050],
                    [0.225, 0.225, 0.250, 0.200, 0.100],
                    [0.150, 0.150, 0.100, 0.300, 0.300],
                    [0.275, 0.275, 0.050, 0.150, 0.250],
                ]
            ]
        )
        definitions = automatic_semantic_event_scores(
            torch.zeros(1, 4, 2),
            torch.zeros(1, 4, 2),
            uncertainty,
            margin,
            bridge_logprob=bridge,
            answer_logprob=answer,
        )

        result = semantic_alignment_metrics(memberships, definitions, event_quantile=0.5)

        self.assertFalse(result["modes"]["knowledge_enrichment"]["available"])
        self.assertFalse(result["modes"]["information_routing"]["available"])
        self.assertAlmostEqual(result["modes"]["concept_composition"]["auroc"], 1.0)
        self.assertAlmostEqual(result["modes"]["prediction_refinement"]["auroc"], 0.5)
        self.assertAlmostEqual(
            result["modes"]["prediction_refinement"]["average_precision"], 7.0 / 12.0
        )
        self.assertAlmostEqual(result["modes"]["hop_transition"]["auroc"], 1.0)
        self.assertEqual(result["num_available_modes"], 3)
        self.assertIsNone(result["macro_auroc"])
        self.assertAlmostEqual(
            result["diagnostic_macro_auroc_available_modes"], 5.0 / 6.0
        )
        self.assertAlmostEqual(
            result["diagnostic_macro_average_precision_available_modes"], 31.0 / 36.0
        )
        json.dumps(result, allow_nan=False)

    def test_rollout_error_matches_mean_squared_l2_definition(self) -> None:
        target = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [7.0, 0.0, 0.0, 0.0]]]
        )
        predicted = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]]
        )
        dimensions = {"z": 1, "concept": 1, "belief": 1, "uncertainty": 1}

        result = rollout_metrics(predicted, target, dimensions)

        self.assertAlmostEqual(result["rollout_error"], 5.0)
        self.assertAlmostEqual(result["mse"], 1.25)
        self.assertAlmostEqual(result["per_horizon"][0]["mse"], 0.25)
        self.assertAlmostEqual(result["per_horizon"][1]["mse"], 2.25)
        self.assertAlmostEqual(result["final_state_mse"], 2.25)
        self.assertEqual(result["positive_r2_horizon"], 0)


if __name__ == "__main__":
    unittest.main()
