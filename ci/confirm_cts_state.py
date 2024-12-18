"""
Attempts building & running all CTS category targets in an existing build directory and compares
their passing / failing with the info from `cts_state.csv`. If the two sources differ, reports
their disparities and exits with a non-zero code.
"""

from argparse import ArgumentParser
from operator import itemgetter
import os
import subprocess
import sys

import pandas as pd

parser = ArgumentParser()
parser.add_argument('cts_root', type=str, help='SYCL-CTS repository path')
parser.add_argument('cts_build_dir', type=str, help='SYCL-CTS + SimSYCL build directory')
args = parser.parse_args()
cts_root = os.path.realpath(args.cts_root)
cts_build_dir = os.path.realpath(args.cts_build_dir)

state_file = pd.read_csv('ci/cts_state.csv', delimiter=';')
tests_in_state_file = set(state_file['suite'])

tests_dir = os.path.join(cts_root, 'tests')
tests_in_cts = set(t for t in os.listdir(tests_dir)
                   if os.path.isdir(os.path.join(tests_dir, t))
                   and t not in ['common', 'extension'])

n_build_failed = 0
n_run_failed = 0
n_passed = 0
changed = []
for test in sorted(tests_in_cts):
    status_in_state_file = state_file['status'][state_file['suite'] == test].values
    status_in_state_file = status_in_state_file[0] if status_in_state_file.size > 0 else 'not in list'
    if status_in_state_file == 'not applicable': continue

    print('testing', test, end='... ', flush=True)
    r = subprocess.run(['cmake', '--build', cts_build_dir, '--target', f'test_{test}'],
                       cwd=cts_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode == 0:
        r = subprocess.run(os.path.join(cts_build_dir, 'bin', f'test_{test}'), cwd=cts_root,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            status_now = 'passed'
            n_passed += 1
        else:
            status_now = 'run failed'
            n_run_failed += 1
    else:
        status_now = 'build failed'
        n_build_failed += 1

    if status_now == status_in_state_file:
        print(status_now)
    else:
        print(f'{status_now}, but was {status_in_state_file}')
        changed.append((test, status_in_state_file, status_now))

print(f'\n{n_passed} passed, {n_run_failed} failed to run, {n_build_failed} failed to build')

for test in tests_in_state_file - tests_in_cts:
    status_in_state_file = state_file['status'][state_file['suite'] == test].values[0]
    changed.append((test, status_in_state_file, 'not in SYCL-CTS'))

if changed:
    print(f'\n{len(changed)} change(s) compared to cts_state.csv:')
    changed.sort(key=itemgetter(0))
    for test, status_in_state_file, status_now in changed:
        print(f'  - {test}: {status_in_state_file} -> {status_now}')
    sys.exit(1)
