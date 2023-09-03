from jupyter_client import KernelManager
import unittest
import numpy as np
import os
import argparse

# local
from .testing_helpers import *


class NotebookTest(unittest.TestCase):
    test_file = None
    module = None

    @classmethod
    def setUp(self):
        self.configs = fetch_notebook_configs(self.module)
        self.km = KernelManager()
        self.km.start_kernel(
            extra_arguments=["--pylab=inline"], stderr=open(os.devnull, "w")
        )
        self.kc = self.km.blocking_client()
        self.kc.start_channels()
        self.kc.execute_interactive("import os;os.environ['IVY_ROOT']='.ivy'")

    @classmethod
    def tearDown(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel()
        del self.km

    def _sanity_checks(self, res, gt_res, execution_count):

        self.assertEqual(execution_count, res["execution_count"], "Asynchronous execution failed !")

        if res['output_type'] in ('pyerror', 'error'):
            res_text = f"runtime output throws an error -: " \
                       f"{res['ename']}\n with value -: {res['evalue']}" \
                       f" and traceback-:\n{res['traceback']}\n"
        else:
            res_text = f"runtime output {res['text']}"

        self.assertEqual(
            res['output_type'], gt_res['output_type'],
            f"{res_text} does not match "
            f" ground truth output\n"
            f"{gt_res['text'] if hasattr(gt_res, 'text') else gt_res['data']}")

    def _benchmarking_test(self, data):
        # Todo: current design assumes we run benchmarking only for two cells per notebook

        speedup_gt = benchmarking_helper(data[0]['ground_truth_result'], data[1]['ground_truth_result'])
        speedup_runtime = benchmarking_helper(data[0]['execution_result'], data[1]['execution_result'])

        # gt timedelta should not be greater than execution time delta
        self.assertLessEqual(speedup_gt, speedup_runtime)

    @staticmethod
    def _assert_all_close(rt_tensor, gt_tensor, rtol=1e-05, atol=1e-08):
        assert np.allclose(
            np.nan_to_num(rt_tensor), np.nan_to_num(gt_tensor), rtol=rtol, atol=atol
        ), (
            f" the results from notebook "
            f"and runtime "
            f"do not match\n {rt_tensor}!={gt_tensor} \n\n"
        )


    def _test_cell(self, test_out, cell_out, execution_count, value_test=True):
        # ToDo: make it more generic
        # occasionally there can be multiple streams of data
        if len(test_out) > len(cell_out):
            concatenate_outs(test_out)

        for res, gt_res in zip(test_out, cell_out):
            # smoke tests
            self._sanity_checks(res, gt_res, execution_count)
            # value test
            if value_test:
                if hasattr(gt_res, "data"):
                    process_display_data(None, gt_res)

                arrays_to_test = value_test_helper(res['text'], gt_res['text'])

                if arrays_to_test:
                    self._assert_all_close(arrays_to_test[0], arrays_to_test[1])

                else:
                    self.assertEqual(
                        res['text'], gt_res['text'],
                        f"runtime output {res['text']} does not match "
                        f"the ground truth output {gt_res['text']}")



    def test_notebook(self):
        test_configs = fetch_notebook_configs('03_compile_code.ipynb')
        test_buffer = Buffer()
        test_file = fetch_nb('03_compile_code.ipynb', 'basics')
        for cell in test_file.cells:
            outs = []
            if cell.cell_type != "code":
                continue
            if "pip install" in cell.source:
                continue
            try:
                self.kc.execute_interactive(
                    cell.source,
                    output_hook=lambda msg: record_output(
                        msg, outs, cell.execution_count
                    ),
                    timeout=500,
                )
            except Exception as e:
                print("failed to run cell:", repr(e))
                print(cell.source)
                continue

            # if cell.execution_count in test_configs.get('cell_numbers'):
            #     test_configs.update({"res": outs, "gt_res": cell.outputs, "execution_count": cell.execution_count})
            #     test_buffer.set_data(test_configs)
            #     continue

            # standard tests
            with self.subTest(msg=f"Testing cell {cell.execution_count}"):
                # Todo: make _test_cell more generic
                self._test_cell(outs, cell.outputs, cell.execution_count)

        # special cases
        for test_type, data in test_buffer.get_data():
            with self.subTest(msg=f"Run {test_type}"):
                # Todo: this should be inside _test_cell
                self._benchmarking_test(data)


class IterativeTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return IterativeTestResult(self.stream, self.descriptions, self.verbosity)


class IterativeTestResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.stream.writeln(f"Running test: {test.id()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook_path", help="Path to the notebook file")
    parser.add_argument("module", help="Can either test examples or Basics")
    args = parser.parse_args()

    unittest.main(testRunner=IterativeTestRunner)

    # if result.wasSuccessful():
    #     exit(0)  # Tests passed
    # else:
    #     exit(1)  # Tests failed
