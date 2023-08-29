from typing import *

configs = {
  "module": {
    "03_compile_code.ipynb":
            {
              "cell_numbers": (11, 12),
              "run": "benchmarking_test",
              "test_fn": lambda fn: None,
              "tolerance": Union[int, float],
              "skip_standard_test": True,
              "action": "Run benchmarking tests to compare speedups dynamically",
            },
    }
}
