from typing import *

configs = {
  "module": {
    "03_compile_code.ipynb":
            {
              "cell_numbers": (11, 12),
              "run": "Benchmarking Test",
              "regex": r'([\d.]+) ([µmns]+)',
              "tolerance": Union[int, float],
              "action": "Run benchmarking tests to compare speedups dynamically",
            },
    }
}
