"""Regression tests: REMORA is a traversal scorer in ALL five experiments."""
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('scoring_contract', HERE / 'scoring_contract.py')
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class TraversalContractTests(unittest.TestCase):
    def test_default_and_formal_config_are_traversal(self):
        for name in ('config.example.json', 'config.native100-exp1-nvme.json', 'config.native100-gpu.json'):
            config = json.loads((HERE / name).read_text())
            contract.require_traversal(config)

    def test_terminal_and_missing_placement_are_rejected(self):
        for placement in ('terminal', None):
            with self.assertRaises(ValueError):
                contract.require_traversal({'scoring_placement': placement})

    def test_every_ablation_and_rate_uses_traversal_contract(self):
        for variant in ('remora', 'euclidean', 'alignment', 'balanced', 'centered'):
            for bits in (16, 32, 48, 64, 96, 128):
                with self.subTest(variant=variant, bits=bits):
                    command = ['--ji-model', 'model.bin', '--ji-side-codes', 'codes.bin', '--ji-query-space', 'raw']
                    contract.validate_command(command, variant)
                    with self.assertRaises(ValueError):
                        contract.validate_command(command + ['--terminal-ji'], variant)

    def test_summary_verifies_runtime_not_just_config(self):
        summary = {'ji_traversal_enabled': True, 'terminal_ji': {'enabled': False},
                   'ji_query_space': 'raw', 'mode': 'ji-pure-rerank'}
        contract.validate_native_summary(summary, 'remora')
        for changes in ({'ji_traversal_enabled': False}, {'ji_query_space': 'scorer'},
                        {'terminal_ji': {'enabled': True}}, {'mode': 'primary-traversal-terminal-ji-rerank'}):
            with self.assertRaises(ValueError):
                contract.validate_native_summary(dict(summary, **changes), 'remora')

    def test_baseline_stays_primary(self):
        contract.validate_command(['--disable-ji'], 'primary')
        contract.validate_native_summary({'ji_traversal_enabled': False, 'terminal_ji': {'enabled': False}}, 'primary')
        with self.assertRaises(ValueError):
            contract.validate_command([], 'primary')


if __name__ == '__main__':
    unittest.main()
