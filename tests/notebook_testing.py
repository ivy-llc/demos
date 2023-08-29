from jupyter_client import KernelManager
import unittest
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

    def _benchmarking_test(self, case):
        pass

    def _test_cell(self, test_out, cell_out, execution_count, value_test=True):
        for res, gt_res in zip(test_out, cell_out):

            # smoke tests
            self.assertEqual(execution_count, res["execution_count"])

            self.assertEqual(res['output_type'], gt_res['output_type'])

            if hasattr(gt_res, "name") and getattr(gt_res, "name") == "stderr":
                continue

            # value test
            if value_test:
                if hasattr(gt_res, "data"):
                    process_display_data(None, gt_res)

                self.assertEqual(
                    sanitize(res["text"]),
                    sanitize(gt_res["text"]),
                    (
                        f"Cell output {res['text']} does not match ground truth output"
                        f" {gt_res['text']} for cell number {execution_count}"
                    ),
                )

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
                )
            except Exception as e:
                print("failed to run cell:", repr(e))
                print(cell.source)
                continue

            if cell.execution_count in test_configs.get('cell_numbers'):
                test_configs.update({"res": outs, "gt_res": cell.outputs, "execution_count": cell.execution_count})
                test_buffer.set_data(test_configs)
                continue


            # standard tests
            with self.subTest(msg=f"Testing cell {cell.execution_count}"):
                #Todo: make _test_cell more generic
                self._test_cell(outs, cell.outputs, cell.execution_count)

        # special cases
        for cases in test_buffer.get_data():
            with self.subTest(msg=f"Run {cases['type_of_test']} for cells {cases['execution_count']}"):
                #Todo: this should be inside _test_cell
                self._benchmarking_test(cases)



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