# Copyright 2025 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Print metrics from ncu-rep file.

Usage:
  ncu_rep -i <ncu-rep-file> [metrics|kernels|value]
    [-f <format>] [-k <kernel name>]
    [-m metric1] [-m metric2]
  metrics: print all metric names
  kernels: print all kernel names
  value (default): print values of metrics as in -m
"""

from collections.abc import Sequence
import csv
import json
import logging
import shutil
import subprocess
import sys
from typing import TextIO
from absl import app
from absl import flags

_INPUT_FILE = flags.DEFINE_string(
    "i", None, "Input .ncu-rep file", required=False
)
_METRICS = flags.DEFINE_multi_string(
    "m",
    [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.max",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "launch__registers_per_thread",
    ],
    "Input .ncu-rep file",
)
_FORMAT = flags.DEFINE_enum(
    "f",
    "md",
    ["md", "csv", "json", "raw"],
    "Output format: md (default), csv, or json",
)
_KERNEL = flags.DEFINE_string(
    "k",
    None,
    "kernel to print (prints first kernel if empty)",
)

ncu_bin = shutil.which("ncu")
if not ncu_bin:
  ncu_bin = "/usr/local/cuda/bin/ncu"
logging.info("ncu binary: %s", ncu_bin)


def get_metrics_by_kernel(
    rows: list[list[str]],
) -> dict[str, dict[str, tuple[str, str]]]:
  """Converts ncu-rep table to a dictionary of metrics by kernel.

  Args:
    rows: ncu-rep table rows

  Returns:
    dictionary of metrics by kernel
  """
  name_index = {}
  units = rows[1]
  for i, name in enumerate(rows[0]):
    name_index[name] = i
  results = {}
  for kernel in rows[2:]:
    values = {}
    for idx, name in enumerate(rows[0]):
      values[name] = (kernel[idx], units[idx])
    kernel_name = values["Kernel Name"][0]
    results[kernel_name] = values
  return results


def get_kernel_metrics_rows(
    metrics: list[str],
    all_metrics: dict[str, dict[str, tuple[str, str]]],
    kernel_name: str,
) -> list[list[str]]:
  """Returns the metrics to print for the given kernel.

  Args:
    metrics: list of metrics names to print
    all_metrics: dictionary of metrics by kernel, extracted from ncu-rep table
    kernel_name: kernel name to print, returns first kernel if empty

  Returns:
    list of rows [name, value, unit] per metric.
  """
  if not all_metrics:
    raise app.UsageError("no metrics found")
  for kernel, vals in all_metrics.items():
    if kernel_name and kernel != kernel_name:
      continue
    result = []
    for name in metrics:
      if name not in vals:
        raise app.UsageError(f"metric '{name}' is not found")
      result.append([name, vals[name][0], vals[name][1]])
    return result
  raise app.UsageError(f"kernel '{kernel_name}' is not found")


def write_metrics_markdown(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in markdown."""
  name_width = max(len(m[0]) for m in metrics)
  value_width = max(max(len(m[1]) for m in metrics), len("value"))
  unit_width = max(max(len(m[2]) for m in metrics), len("unit"))
  out.write(
      f"{'Metric'.ljust(name_width)} | {'Value'.rjust(value_width)} | Unit\n"
  )
  out.write(
      f"{'-' * name_width }-|-{'-' * value_width }-|-{'-' * unit_width }\n"
  )
  for name, value, unit in metrics:
    out.write(
        f"{name.ljust(name_width)} | {value.rjust(value_width)} | {unit}\n"
    )


def write_metrics_csv(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in csv."""
  writer = csv.writer(out, lineterminator="\n")
  writer.writerow(["metric", "value", "unit"])
  writer.writerows(metrics)


def write_metrics_json(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in JSON."""
  data = {}
  for name, value, unit in metrics:
    data[name] = {"value": value, "unit": unit}
  json.dump(data, out, sort_keys=True)
  out.write("\n")


def write_metrics_raw(out: TextIO, metrics: list[list[str]]):
  """Formats metrics in raw."""
  for _, value, unit in metrics:
    out.write(f"{value} {unit}\n")


def main(argv: Sequence[str]) -> None:
  input_name = _INPUT_FILE.value
  if not input_name:
    # We can't use required=True due to unit tests.
    raise app.UsageError("input file (-i) is required")
  cmd = [ncu_bin, "-i", input_name, "--csv", "--page", "raw"]
  out = subprocess.check_output(cmd, text=True).strip()
  rows = list(csv.reader(out.splitlines()))
  name_index = {}
  for i, name in enumerate(rows[0]):
    name_index[name] = i

  op = argv[1] if len(argv) > 1 else "value"
  if op == "metrics":
    for name in rows[0]:
      print(name)
    return

  metrics_by_kernel = get_metrics_by_kernel(rows)

  if op == "kernels":
    for name in metrics_by_kernel:
      print(name)
    return

  metrics = get_kernel_metrics_rows(
      _METRICS.value, metrics_by_kernel, _KERNEL.value
  )

  fmt = _FORMAT.value
  if fmt == "csv":
    write_metrics_csv(sys.stdout, metrics)
  elif fmt == "json":
    write_metrics_json(sys.stdout, metrics)
  elif fmt == "raw":
    write_metrics_raw(sys.stdout, metrics)
  else:
    write_metrics_markdown(sys.stdout, metrics)


if __name__ == "__main__":
  app.run(main)
