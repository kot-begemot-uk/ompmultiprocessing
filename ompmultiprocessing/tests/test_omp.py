#!/usr/bin/python3


'''BPF Map functional test
'''

# pybpfmap, Copyright (c) 2023 RedHat Inc
# pybpfmap, Copyright (c) 2023 Cambridge Greys Ltd

# This source code is licensed under both the BSD-style license (found in the
# LICENSE file in the root directory of this source tree) and the GPLv2 (found
# in the COPYING file in the root directory of this source tree).
# You may select, at your option, one of the above-listed licenses.


from ompmultiprocessing import OMPProcessManager, OMPStrategy

import os

SIMPLE_MAP="simple.json"
NUMA_MAP="numa.json"

def runnable(arg, expected_places=None, expected_threads=0):

    assert(expected_places == os.environ["OMP_PLACES"])
    assert(expected_threads == int(os.environ["OMP_NUM_THREADS"]))

def set_vllm_environment(env=None):
    if env is None:
        try:
            del os.environ["VLLM_CPU_OMP_THREADS_BIND"]
        except KeyError:
            pass
    else:
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = env

def test_01_simple():
    '''Test Simple Topology'''
    set_vllm_environment()
    kwargs = dict()
    manager = OMPProcessManager(mock="simple.json", affinity=set(range(0,11)))
    kwargs["expected_places"] = str(set(range(0,6)))
    kwargs["expected_threads"] = 6
    manager.run(runnable, None, **kwargs)

def test_02_complex():
    '''Test Complex Topology'''
    set_vllm_environment()
    kwargs = dict()
    manager = OMPProcessManager(mock="numa.json", affinity=set(range(0,96)))
    kwargs["expected_places"] = str(set(range(64,96)))
    kwargs["expected_threads"] = 32
    manager.run(runnable, None, **kwargs)
    kwargs["expected_places"] = str(set(range(32,64)))
    kwargs["expected_threads"] = 32
    manager.run(runnable, None, **kwargs)
    kwargs["expected_places"] = str(set(range(0,32)))
    kwargs["expected_threads"] = 32
    manager.run(runnable, None, **kwargs)

def test_03_mask():
    '''Test Complex Topology with a mask'''
    set_vllm_environment()
    kwargs = dict()
    manager = OMPProcessManager(mock="numa.json", affinity=set(range(32,96)))
    assert(len(manager.omp_places) == 2)
    kwargs["expected_places"] = str(set(range(64,96)))
    kwargs["expected_threads"] = 32
    manager.run(runnable, None, **kwargs)
    kwargs["expected_places"] = str(set(range(32,64)))
    kwargs["expected_threads"] = 32
    manager.run(runnable, None, **kwargs)

def test_04_vllm_environment():
    '''Test VLLM topology environment variables'''
    kwargs = dict()
    manager = OMPProcessManager(global_mask="32-63|64-95", mock="numa.json", affinity=set(range(0,96)))
    assert(len(manager.omp_places) == 2)
    kwargs["expected_places"] = str(set(range(64,96)))
    kwargs["expected_threads"] = 32
    manager.run(runnable, None, **kwargs)
    kwargs["expected_places"] = str(set(range(32,64)))
    kwargs["expected_threads"] = 32
    manager.run(runnable, None, **kwargs)

def test_05_vllm_environment_conflict():
    '''Test VLLM topology environment variables'''
    kwargs = dict()
    manager = OMPProcessManager(global_mask="32-63|64-95", mock="numa.json", affinity=set(range(0,64)))
    assert(len(manager.omp_places) == 1)
    kwargs["expected_places"] = str(set(range(32,64)))
    kwargs["expected_threads"] = 32
    manager.run(runnable, None, **kwargs)

def test_06_smt():
    '''Test Simple Topology with smt 2'''
    set_vllm_environment()
    kwargs = dict()
    manager = OMPProcessManager(mock="simple.json", affinity=set(range(0,12)), strategy=OMPStrategy(smt=2))
    kwargs["expected_places"] = str(set(range(0,12)))
    kwargs["expected_threads"] = 12
    manager.run(runnable, None, **kwargs)

def test_07_split():
    '''Test Simple Topology split in 2'''
    set_vllm_environment()
    kwargs = dict()
    manager = OMPProcessManager(mock="simple.json", affinity=set(range(0,12)), strategy=OMPStrategy(split=2))
    assert(len(manager.omp_places) == 2)
    kwargs["expected_places"] = str(set(range(3,6)))
    kwargs["expected_threads"] = 3
    manager.run(runnable, None, **kwargs)
    kwargs["expected_places"] = str(set(range(0,3)))
    kwargs["expected_threads"] = 3
    manager.run(runnable, None, **kwargs)

def test_08_reserve_1():
    '''Test Reservation'''
    set_vllm_environment()
    kwargs = dict()
    manager = OMPProcessManager(mock="simple.json", affinity=set(range(0,12)), strategy=OMPStrategy(reserve=1))
    assert(len(manager.omp_places) == 1)
    kwargs["expected_places"] = str(set(range(0,5)))
    kwargs["expected_threads"] = 5
    manager.run(runnable, None, **kwargs)
