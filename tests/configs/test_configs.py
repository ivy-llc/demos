from typing import *

configs = {
    "module": {
        "03_compile_code.ipynb":
            {
                "cell_numbers": (11, 12),
                "run": "benchmarking_test",
                "regex": r'([\d.]+) ([µmns]+)',
                "tolerance": Union[int, float],
                "action": "Run benchmarking tests in the next iteration to compare speedups dynamically",
            },
        "07_transpile_any_library.ipynb":
            {
                "cell_numbers": (4, 6),
                "run": "skip_test",
                "action": "No testing setup for images yet !"
            },
        "alexnet_demo.ipynb":
            {
                "cell_numbers": (4, 18),
                "run": "skip_test",
                "action": "No testing setup for images yet !"
            },
        "image_segmentation_with_ivy_unet.ipynb":
            {
                "cell_numbers": (5, 10, 11, 13, 16),
                "run": "skip_test",
                "action": "No testing setup for images yet !"
            },
        "mmpretrain_to_jax.ipynb":
            {
                "cell_numbers": (13, 14),
                "run": "benchmarking_test",
                "regex": r'([\d.]+) ([µmns]+)',
                "tolerance": Union[int, float],
                "action": "Run benchmarking tests in the next iteration to compare speedups dynamically",
            },
        "resnet_demo.ipynb":
            {
                "cell_numbers": 5,
                "run": "skip_test",
                "action": "No testing setup for images yet !"

            },
        "torch_to_jax.ipynb":
            {
                "cell_numbers": (15, 16),
                "run": "benchmarking_test",
                "regex": r'([\d.]+) ([µmns]+)',
                "tolerance": Union[int, float],
                "action": "Run benchmarking tests in the next iteration to compare speedups dynamically",
            }
    }
}
