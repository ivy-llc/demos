import re
import nbformat
from datetime import timedelta

from .configs.test_configs import *


class OutputMessage:
    def __init__(self):
        self.output_type = None
        self.text = None
        self.execution_count = None

    def set_error_attributes(self, attrs):
        self.ename = attrs["ename"]
        self.evalue = attrs["evalue"]
        self.traceback = attrs["traceback"]

    def as_dict(self):
        return vars(self)


def process_stream_content(out, content):
    setattr(out, "name", content.get("name"))
    out.text = content["text"]


def process_error(out, err_attrs):
    out.set_error_attributes(err_attrs)


def process_display_data(out, content):
    for mime, data in content["data"].items():
        mime_type = (
            mime.split("/")[-1].lower().replace("+xml", "").replace("plain", "text")
        )
        target_obj = out if out is not None else content
        setattr(target_obj, mime_type, data)


def record_output(msg, outs, execution_count):
    msg_type = msg["header"]["msg_type"]
    content = msg["content"]

    processing_functions = {
        "stream": process_stream_content,
        "display_data": process_display_data,
        "pyout": process_display_data,
        "execute_result": process_display_data,
        "pyerr": process_error,
        "error": process_error,
    }

    out = OutputMessage()
    processing_function = processing_functions.get(msg_type)
    if processing_function:
        processing_function(out, content)
        out.output_type = msg_type
        out.execution_count = execution_count
        outs.append(out.as_dict())


class Buffer:
    def __init__(self):
        self.buffers = {}

    @staticmethod
    def _parse_configs(test_configs):
        gt = test_configs.get('gt_res', [])
        res = test_configs.get('res', [])

        processed_results = []
        for idx, (execution_result, ground_truth_result) in enumerate(zip(res, gt)):
            if hasattr(ground_truth_result, 'data'):
                process_display_data(None, ground_truth_result)

            execution_text = execution_result.get('text', 'No result')
            ground_truth_text = ground_truth_result.get('text', 'No ground truth')

            # ToDo: shouldn't be done here, run standard tests once
            # smoke test
            assert execution_result['output_type'] == ground_truth_result['output_type']

            data = {
                "index": idx + 1,
                "cell_number": test_configs['execution_count'],
                "execution_result": execution_text,
                "ground_truth_result": ground_truth_text,
            }
            processed_results.append(data)

        return processed_results

    def set_data(self, test_configs):
        data = Buffer._parse_configs(test_configs)
        type_of_test = test_configs.get('run', 'Unknown')
        if type_of_test not in self.buffers:
            self.buffers[type_of_test] = []
        self.buffers[type_of_test].extend(data)

    def get_data(self):
        return self.buffers.items()


def value_test_helper(s):
    """
    Sanitize a string for comparison.

    fix universal newlines, strip trailing newlines, and normalize
    likely random values (memory addresses and UUIDs)
    """
    # ToDo: currently we're only sanitizing array types

    pattern = r'\[([^\]]+)\]'
    tensor = re.search(pattern, s)
    if tensor:
        return eval(tensor.group())
    return None


import re
from datetime import timedelta


def benchmarking_helper(exec_fn, exec_comp):
    pattern = r'([\d.]+) ([µmns]+)'
    match_1 = re.search(pattern, exec_fn)
    match_2 = re.search(pattern, exec_comp)

    if match_1 and match_2:
        execution_time_1, time_unit_1 = float(match_1.group(1)), match_1.group(2)
        execution_time_2, time_unit_2 = float(match_2.group(1)), match_2.group(2)

        time_unit_mapping = {'µs': 'microseconds', 'ms': 'milliseconds', 'ns': 'nanoseconds'}

        if time_unit_1 in time_unit_mapping and time_unit_2 in time_unit_mapping:
            time_delta_1 = timedelta(**{time_unit_mapping[time_unit_1]: execution_time_1})
            time_delta_2 = timedelta(**{time_unit_mapping[time_unit_2]: execution_time_2})

            speedup = abs(time_delta_1/time_delta_2)
            return speedup

    return None


def fetch_nb(notebook, module):
    path = "/"
    if module == "basics":
        path = "learn_the_basics/"
    elif module == "examples":
        path = "example_and_demos/"
    file = path + notebook
    with open(file) as f:
        nb = nbformat.reads(f.read(), nbformat.current_nbformat)
        return nb

def fetch_notebook_configs(file):
    return configs["module"].get(file, {})
