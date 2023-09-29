from typing import *

configs = {
    "module": {
        "03_compile_code.ipynb":
            {
                "cell_numbers": (11, 12),
                "run": "Benchmarking test",
                "regex": r'([\d.]+) ([µmns]+)',
                "tolerance": Union[int, float],
                "action": "Run benchmarking tests in the next iteration to compare speedups dynamically",
            },
        "07_transpile_any_library.ipynb":
            {
                "cell_numbers": (4, 6),
                "run": "Skip test",
                "action": "No testing setup for images yet !"
            },
    }
}
