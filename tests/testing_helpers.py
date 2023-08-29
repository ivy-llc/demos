import re
import nbformat

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
        self.buffers = []

    @staticmethod
    def _parse_configs(test_configs):
        gt = test_configs.get('gt_res', [])
        res = test_configs.get('res', [])

        processed_results = []
        for idx, (execution_result, ground_truth_result) in enumerate(zip(res, gt)):
            if hasattr(ground_truth_result, 'data'):
                process_display_data(None, ground_truth_result)

            type_of_test = test_configs.get('run', 'Unknown')
            execution_text = execution_result.get('text', 'No result')
            ground_truth_text = ground_truth_result.get('text', 'No ground truth')

            data = {
                "index": idx+1,
                "type_of_test": type_of_test,
                "execution_result": execution_text,
                "ground_truth_result": ground_truth_text,
                "cell_number": test_configs['execution_count'],
            }
            processed_results.append(data)

        return processed_results

    def set_data(self, test_configs):
        data = Buffer._parse_configs(test_configs)
        self.buffers.extend(data)

    def get_data(self):
        return self.buffers



def sanitize(s):
    """Sanitize a string for comparison.

    fix universal newlines, strip trailing newlines, and normalize
    likely random values (memory addresses and UUIDs)
    """
    if not isinstance(s, str):
        return s
    # normalize newline:
    s = s.replace("\r\n", "\n")

    # ignore trailing newlines (but not space)
    s = s.rstrip("\n")

    # normalize hex addresses:
    s = re.sub(r"0x[a-f0-9]+", "0xFFFFFFFF", s)

    # normalize UUIDs:
    s = re.sub(r"[a-f0-9]{8}(\-[a-f0-9]{4}){3}\-[a-f0-9]{12}", "U-U-I-D", s)
    return s


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
