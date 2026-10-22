"""No-fit qualification of isolated Figure3g correction wiring."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[2] / 'cebra_replica/monkey/complete_figure3g_position_only.py'
spec = importlib.util.spec_from_file_location('figure3g_position_only_correction', PATH)
correction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correction)


class Figure3gCorrectionTests(unittest.TestCase):
    def test_actual_preflight_exact_config_and_separate_paths(self):
        plan = correction.plan()
        self.assertEqual(plan['item'][1:], ['all', 'position', 4])
        self.assertEqual(plan['config']['seed'], 42)
        self.assertEqual(plan['config']['max_iterations'], 10000)
        self.assertEqual(plan['config']['label_mode'], 'position')
        self.assertEqual(plan['dry_run_optimizer_updates'], 0)
        self.assertEqual(plan['label_shape'], [218400, 2])
        self.assertNotIn(correction.OLD, correction.OUTPUT.parents)

    def test_scoped_reuse_wiring_and_restoration(self):
        old = (correction.fit.ROOT, correction.export.ROOT, correction.plot.FIGURES)
        def fitted(item):
            self.assertEqual(item, correction.ITEM)
            self.assertEqual(correction.fit.ROOT, correction.OUTPUT / 'stage01_fit')
            return 'mock_fit_no_optimizer'
        with mock.patch.object(correction.fit, 'fit_one', side_effect=fitted) as fit, mock.patch.object(correction.export, 'export_one', return_value='mock_export') as export, mock.patch.object(correction.plot, 'plot_3efg') as render:
            correction.fit_export_render()
            fit.assert_called_once_with(correction.ITEM)
            export.assert_called_once_with(correction.ITEM)
            self.assertEqual(render.call_args.args[0], correction.ITEM[0])
            self.assertEqual(render.call_args.args[3:5], ((0, 1, 2), 'row'))
        self.assertEqual(old, (correction.fit.ROOT, correction.export.ROOT, correction.plot.FIGURES))

    def test_restore_bindings_on_error(self):
        original = correction.fit.ROOT
        with self.assertRaisesRegex(RuntimeError, 'mock failure'):
            with correction.scoped_paths():
                raise RuntimeError('mock failure')
        self.assertEqual(correction.fit.ROOT, original)

    def test_prerequisite_requires_all144_complete_and_no_lock(self):
        state = dict(status='complete', planned=144, completed=144,
                     records={str(i): {'status': 'complete'} for i in range(144)})
        self.assertTrue(correction.shuffle_ready(state, False))
        self.assertFalse(correction.shuffle_ready(state, True))
        state['records']['10']['status'] = 'fit_complete'
        self.assertFalse(correction.shuffle_ready(state, False))


if __name__ == '__main__':
    unittest.main()
