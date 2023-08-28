from jupyter_client import KernelManager
import unittest
import os
import argparse

# local
from .testing_helpers import *


class NotebookTest(unittest.TestCase):
    def setUp(self):
        self.km = KernelManager()
        self.km.start_kernel(
            extra_arguments=["--pylab=inline"], stderr=open(os.devnull, "w")
        )
        self.kc = self.km.blocking_client()
        self.kc.start_channels()
        self.kc.execute_interactive("import os;os.environ['IVY_ROOT']='.ivy'")

    def tearDown(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel()
        del self.km

    def _test_cell(self, test_out, cell_out, execution_count, value_test=True):
        for result, gt in zip(test_out, cell_out):
            res = result.as_dict()

            # smoke test
            self.assertEqual(execution_count, res["execution_count"])

            if hasattr(gt, "name") and getattr(gt, "name") == "stderr":
                continue

            if value_test:
                if hasattr(gt, "data"):
                    process_display_data(None, gt)

                self.assertEqual(
                    sanitize(res["text"]),
                    sanitize(gt["text"]),
                    (
                        f"Cell output {res['text']} does not match ground truth output"
                        f" {gt['text']} for cell number {execution_count}"
                    ),
                )

    def test_notebook(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("notebook_path", help="Path to the notebook file")
        parser.add_argument("module", help="Can either test examples or Basics")
        nb = fetch_nb(parser.parse_args())

        for cell in nb.cells:
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

            # Start a subtest for each cell
            with self.subTest(msg=f"Testing cell {cell.execution_count}"):
                self._test_cell(outs, cell.outputs, cell.execution_count)


class IterativeTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return IterativeTestResult(self.stream, self.descriptions, self.verbosity)


class IterativeTestResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.stream.writeln(f"Running test: {test.id()}")


if __name__ == "__main__":
    unittest.main(testRunner=IterativeTestRunner())
