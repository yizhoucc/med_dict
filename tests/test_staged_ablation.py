import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

import staged_ablation as sa
import ult


class VariantConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = yaml.safe_load(
            (sa.PROJECT_ROOT / "exp" / "v32_pdac_annotated_test.yaml").read_text()
        )

    def test_variant_b_has_only_serialization_gates(self):
        config = sa.build_variant_config(self.base, "B", [0, 1])
        sa.validate_variant_config("B", config)
        extraction = config["extraction"]
        self.assertFalse(extraction["verify"])
        self.assertFalse(extraction["inline_postprocessing"])
        self.assertFalse(extraction["post_hooks"])
        self.assertFalse(extraction["tool_calling"])

    def test_variant_c_enables_semantic_gates_but_not_hooks(self):
        config = sa.build_variant_config(self.base, "C", [0, 1])
        sa.validate_variant_config("C", config)
        extraction = config["extraction"]
        self.assertTrue(extraction["verify"])
        self.assertFalse(extraction["inline_postprocessing"])
        self.assertFalse(extraction["post_hooks"])

    def test_variant_d_enables_full_harness(self):
        config = sa.build_variant_config(self.base, "D", [0, 1])
        sa.validate_variant_config("D", config)
        extraction = config["extraction"]
        self.assertTrue(extraction["verify"])
        self.assertTrue(extraction["inline_postprocessing"])
        self.assertTrue(extraction["post_hooks"])

    def test_all_generation_paths_are_greedy(self):
        config = sa.build_variant_config(self.base, "D", [0])
        for generation_config in config["generation"].values():
            self.assertFalse(generation_config["do_sample"])
            self.assertNotIn("temperature", generation_config)
            self.assertNotIn("top_p", generation_config)

    def test_matched_contract_has_exact_pipeline_field_paths(self):
        validation = sa.validate_matched_schema(self.base)
        self.assertTrue(validation["exact_field_path_match"])
        self.assertEqual(validation["field_count"], 31)


class JudgeExportTest(unittest.TestCase):
    def make_study(self, root: Path):
        for variant in sa.VARIANTS:
            variant_dir = root / variant
            variant_dir.mkdir()
            (variant_dir / "manifest.json").write_text(json.dumps({"cancer": "pdac"}))
            with (variant_dir / "outputs.jsonl").open("w") as handle:
                for row_index in (0, 1):
                    handle.write(
                        json.dumps(
                            {
                                "row_index": row_index,
                                "coral_idx": f"p{row_index}",
                                "note_text": f"note {row_index}",
                                "keypoints": {"field": f"{variant}-{row_index}"},
                            }
                        )
                        + "\n"
                    )

    def test_blinding_is_reproducible_and_mapping_is_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "study"
            root.mkdir()
            self.make_study(root)
            public1, private1 = sa.export_judge(root, Path(temp) / "judge1", seed=17)
            public2, private2 = sa.export_judge(root, Path(temp) / "judge2", seed=17)
            self.assertEqual(public1.read_text(), public2.read_text())
            self.assertEqual(private1.read_text(), private2.read_text())
            public_record = json.loads(public1.read_text().splitlines()[0])
            private_record = json.loads(private1.read_text().splitlines()[0])
            self.assertNotIn("variant", json.dumps(public_record).lower())
            self.assertIn("output_a_variant", private_record)
            self.assertEqual(len(public1.read_text().splitlines()), 6)


class PipelineSwitchTest(unittest.TestCase):
    class Tokenizer:
        eos_token_id = 0

    def run_extraction(self, *, verify, enable_postprocessing, answers):
        with mock.patch.object(
            ult,
            "run_model_with_cache_manual",
            side_effect=[(answer, None) for answer in answers],
        ) as generate:
            result = ult.extract_and_verify_v2(
                {
                    "Current_Medications": (
                        "Respond only with JSON using this schema: "
                        '{"current_meds": "active anticancer medication"}'
                    )
                },
                model=object(),
                tokenizer=self.Tokenizer(),
                gen_config={"max_new_tokens": 32, "do_sample": False},
                base_cache=None,
                verify=verify,
                oncology_whitelist={"tamoxifen"},
                supportive_whitelist=set(),
                enable_postprocessing=enable_postprocessing,
            )
        return result, generate.call_count

    def test_b_keeps_raw_field_output_and_skips_semantic_gates(self):
        result, calls = self.run_extraction(
            verify=False,
            enable_postprocessing=False,
            answers=['{"current_meds": "aspirin"}'],
        )
        self.assertEqual(result["Current_Medications"]["current_meds"], "aspirin")
        self.assertEqual(calls, 1)

    def test_c_runs_semantic_gates_without_deterministic_filter(self):
        answer = '{"current_meds": "aspirin"}'
        result, calls = self.run_extraction(
            verify=True,
            enable_postprocessing=False,
            answers=[answer, answer, answer],
        )
        self.assertEqual(result["Current_Medications"]["current_meds"], "aspirin")
        self.assertEqual(calls, 3)

    def test_d_applies_deterministic_filter(self):
        result, calls = self.run_extraction(
            verify=False,
            enable_postprocessing=True,
            answers=['{"current_meds": "aspirin"}'],
        )
        self.assertEqual(result["Current_Medications"]["current_meds"], "")
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
