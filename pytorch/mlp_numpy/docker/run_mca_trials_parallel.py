#!/usr/bin/env python
# coding: utf-8
import os
from subprocess import Popen
from absl import flags, app

flags.DEFINE_integer('n_trials', 30, 'Number of MCA trials to run in parallel')
flags.DEFINE_integer('n_trials_parallel', 10, 'Number of MCA trials to run in parallel')
FLAGS = flags.FLAGS


def run_cmd_async(cmd):
    return Popen(cmd)

def wait_for_all(processes):
    for proc in processes:
        proc.wait()

def main(argv):
    os.environ['VFC_BACKENDS'] = "libinterflop_mca.so"

    cmd = ['python', 'predict_nlp_numpy.py', '--predictions_path']
    processes = []
    for trial_num in range(FLAGS.n_trials):
        results_path = f'/mca_predictions/predictions_{trial_num}.pt'
        proc = run_cmd_async(cmd + [results_path] + argv[1:])
        processes.append(proc)
        if (trial_num + 1 ) % FLAGS.n_trials_parallel == 0:
            wait_for_all(processes)
            processes = []
    wait_for_all(processes)


if __name__ == '__main__':
    app.run(main)


