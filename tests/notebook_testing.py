#!/usr/bin/env python
"""
simple example script for running and testing notebooks.
Usage: `notebook_testing.py foo.ipynb [bar.ipynb [...]]`
Each cell is submitted to the kernel, and the outputs are compared with those stored in the notebook.
Tested with python 3.6 and jupyter 5.0
"""
# License: MIT, but credit is nice (Min RK, ociule).
import os, sys, time
import base64
import re

from collections import defaultdict
from queue import Empty

try:
    # from IPython.kernel import KernelManager
    from jupyter_client import KernelManager
except ImportError:
    print("FAILED: from IPython.kernel import KernelManager")
    from IPython.zmq.blockingkernelmanager import BlockingKernelManager as KernelManager

import nbformat


def compare_png(a64, b64):
    """compare two b64 PNGs (incomplete)"""
    try:
        import Image
    except ImportError:
        pass
    adata = base64.decodestring(a64)
    bdata = base64.decodestring(b64)
    return True


def sanitize(s):
    """sanitize a string for comparison.
    fix universal newlines, strip trailing newlines, and normalize likely random values (memory addresses and UUIDs)
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


def consolidate_outputs(outputs):
    """consolidate outputs into a summary dict (incomplete)"""
    data = defaultdict(list)
    data["stdout"] = ""
    data["stderr"] = ""

    for out in outputs:
        if out.type == "stream":
            data[out.stream] += out.text
        elif out.type == "pyerr":
            data["pyerr"] = dict(ename=out.ename, evalue=out.evalue)
        else:
            for key in (
                "png",
                "svg",
                "latex",
                "html",
                "javascript",
                "text",
                "jpeg",
            ):
                if key in out:
                    data[key].append(out[key])
    return data


def compare_outputs(
    test, ref, skip_compare=("png", "traceback", "latex", "prompt_number")
):
    missing = False
    mismatch = False
    for key in ref:
        if key not in test:
            print(f"missing key: '{key}' not in {test}")
            missing = True
        elif key not in skip_compare and sanitize(test[key]) != sanitize(ref[key]):
            print(f"mismatch '{key}':")
            print(test[key])
            print("  !=  ")
            print(ref[key])
            mismatch = True
    if missing or mismatch:
        return False
    return True


def run_cell(shell, iopub, cell, kc):
    # print cell.source
    # shell.execute(cell.source)
    kc.execute(cell.source)
    # wait for finish, maximum 20s
    shell.get_msg(timeout=300)  # was 20
    outs = []

    while True:
        try:
            msg = iopub.get_msg(timeout=0.2)
        except Empty:
            break
        msg_type = msg["msg_type"]
        if msg_type in ("status", "execute_input"):
            continue
        elif msg_type == "clear_output":
            outs = []
            continue

        content = msg["content"]
        # print msg_type, content
        out = nbformat.NotebookNode(output_type=msg_type)

        if msg_type == "stream":
            out.stream = content["name"]
            out.text = content["text"]
            out.data = content["text"]
            out.name = content["name"]
        elif msg_type in ("display_data", "pyout", "execute_result"):
            out["metadata"] = content["metadata"]
            for mime, data in content["data"].items():
                attr = mime.split("/")[-1].lower()
                # this gets most right, but fix svg+html, plain
                attr = attr.replace("+xml", "").replace("plain", "text")
                setattr(out, attr, data)
            out.data = content["data"]

            if msg_type in ("execute_result", "pyout"):
                out.execution_count = content["execution_count"]
        elif msg_type in ("pyerr", "error"):
            out.ename = content["ename"]
            out.evalue = content["evalue"]
            out.traceback = content["traceback"]
        else:
            print("unhandled iopub msg:", msg_type)

        outs.append(out)
    return outs


def test_notebook(nb):
    km = KernelManager()
    km.start_kernel(extra_arguments=["--pylab=inline"], stderr=open(os.devnull, "w"))
    try:
        kc = km.client()
        kc.start_channels()
        iopub = kc.iopub_channel
    except AttributeError:
        print("AttributeError")
        # IPython 0.13
        kc = km
        kc.start_channels()
        iopub = kc.sub_channel
    shell = kc.shell_channel

    # run %pylab inline, because some notebooks assume this
    # even though they shouldn't
    # shell.execute("pass")
    # kc.execute("pass")

    # kc.execute("import ivy")
    # kc.execute("ivy.set_backend('jax')")
    # kc.execute("ivy.unset_backend()")
    # TODO
    print(os.listdir(os.getcwd()))
    kc.execute("import os;os.environ['IVY_ROOT']='./.ivy'")

    while True:
        try:
            iopub.get_msg(timeout=1)
        except Empty:
            break

    successes = 0
    failures = 0
    errors = 0
    # for ws in nb.worksheets:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "pip install" in cell.source:
            continue
        try:
            outs = run_cell(shell, iopub, cell, kc)
        except Exception as e:
            import pdb

            # pdb.set_trace()
            print("failed to run cell:", repr(e))
            print(cell.source)
            errors += 1
            continue

        failed = False
        # Reverse outputs to filter out initialization warnings
        for out, ref in zip(outs[::-1], cell.outputs[::-1]):
            if not compare_outputs(out, ref):
                failed = True
        if failed:
            failures += 1
        else:
            successes += 1
        sys.stdout.write(".")

    print("tested notebook %s" % nb.metadata.kernelspec.name)
    print("    %3i cells successfully replicated" % successes)
    if failures:
        print("    %3i cells mismatched output" % failures)
    if errors:
        print("    %3i cells failed to complete" % errors)
    kc.stop_channels()
    km.shutdown_kernel()
    del km


if __name__ == "__main__":
    for ipynb in ["learn_the_basics/03_compile_code.ipynb"]:  # sys.argv[1:]:
        print("testing %s" % ipynb)
        with open(ipynb) as f:
            nb = nbformat.reads(f.read(), nbformat.current_nbformat)
        test_notebook(nb)
