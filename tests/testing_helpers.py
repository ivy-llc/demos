import re
import nbformat
import ast
from datetime import timedelta

from configs.test_configs import *


class OutputMessage:
    _set_data = True

    def __init__(self):
        self.output_type = None
        self.text = None
        self.execution_count = None

        # error attributes
        self.traceback = None
        self.evalue = None
        self.ename = None

    def set_error_attributes(self, attrs):
        self.ename = attrs["ename"]
        self.evalue = attrs["evalue"]
        self.traceback = attrs["traceback"]

    def as_dict(self):
        return {key: value for key, value in vars(self).items() if value is not None}

    @property
    def set_data(self):
        return self._set_data


def process_stream_content(out, content):
    # assuming there are no warning messages in the notebook
    if content['name'] == "stderr":
        out._set_data = False
        return
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
        if out.set_data:
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


def _check_for_eval(tensor):
    """
    Sanity check for value test. Replaces
    one or more whitespace/tab characters
    with a comma if it doesn't exist in a
    tensor output.
    Args:
        tensor:

    Returns:

    """
    try:
        tensor = eval(tensor)
    except SyntaxError:
        tensor = eval(re.sub(r'\s+', ', ', tensor))
    return tensor


def check_for_tensors(res, gt_res):
    rt_tensors = []
    gt_tensors = []
    # ToDo: need a more generic regex to parse nested arrays/tensors
    pattern_for_tensors = r'\[([^\]]+)\]+'

    # ToDo: shouldn't be doing it here
    assert len(res) == len(gt_res)     # smoke test

    for a, b in zip(res, gt_res):
        # Extract tensors directly from strings
        rt_tensor = re.search(pattern_for_tensors, a)
        gt_tensor = re.search(pattern_for_tensors, b)

        if rt_tensor:
            # EdgeCase: 04_transpile_code(cell 4) outputs a list w no commas
            rt_tensors.append(_check_for_eval(rt_tensor.group()))
        if gt_tensor:
            # EdgeCase: 04_transpile_code(cell 4) outputs a list w no commas
            gt_tensors.append(_check_for_eval(gt_tensor.group()))

    return rt_tensors, gt_tensors


def value_test_helper(res, gt_res):
    """
    Sanitize a string for comparison.

    fix universal newlines, strip trailing newlines, and normalize
    likely random values (memory addresses and UUIDs)
    """
    # Define a regular expression pattern to match various formats
    pattern = r'(ivy\.array\([^)]*\)|<tf\.Tensor:.*?>|Array\([^)]*\)|tensor\([^)]*\))'

    # Split the input string based on newlines, but not inside matched patterns
    rt_val = re.split(pattern, res)
    gt_val = re.split(pattern, gt_res)

    rt_val, gt_val = check_for_tensors(rt_val, gt_val)

    return (rt_val, gt_val) if rt_val and gt_val else None


def concatenate_outs(rt_outs):
    """
    Handle multiple streams of data
    Args:
        rt_outs:

    Returns:

    """
    for element in rt_outs[1:]:
        rt_outs[0]['text'] += element['text']


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

            speedup = abs(time_delta_1 / time_delta_2)
            return speedup

    return None


def fetch_nb(notebook, module):
    path = "/"
    if module == "basics":
        path = "learn_the_basics/"
    elif module == "examples":
        path = "examples_and_demos/"
    file = path + notebook
    with open(file) as f:
        nb = nbformat.reads(f.read(), nbformat.current_nbformat)
        return nb


def fetch_notebook_configs(file):
    return configs["module"].get(file, {})
