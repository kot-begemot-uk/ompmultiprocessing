'''OMP Aware Multiprocessing drop in replacement for multiprocessing.Process()
Copyright (c) 2026 Red Hat Inc
Copyright (c) 2026 Cambridge Greys Ltd
'''

import multiprocessing
import subprocess
import os
import json



def enumerate_resources(mask = None):
    '''Enumerate system resources'''
    allowed = os.sched_getaffinity(0)
    if mask is not None:
        allowed = allowed & mask
    lscpu = {"cpus":{}, "cores":{}, "nodes":{}}
    for cpu in json.loads(subprocess.run(["lscpu", "-Je"], check=True,
                          capture_output=True).stdout)["cpus"]:
        if cpu["cpu"] in allowed:
            lscpu["cpus"][cpu["cpu"]] = [cpu]
        core = int(cpu["core"])
        if lscpu["cores"].get(core) is None:
            lscpu["cores"][core] = [cpu]
        else:
            lscpu["cores"][core].append(cpu)
        node = int(cpu["node"])
        if lscpu["nodes"].get(node) is None:
            lscpu["nodes"][node] = [cpu]
        else:
            lscpu["nodes"][node].append(cpu)
    return lscpu

def produce_cpu_list(cpus, smt=True):
    '''Produce a CPU list with/without SMT pairs - main cpu list case'''
    mask = []
    for key, value in cpus.items():
        exists = False
        if not smt:
            for cpu in mask:
                if cpu == value[0]["core"]:
                    exists = True
                    break
        if not exists:
            mask.append(key)
    return {"mask":set(mask), "available": True}

def produce_cpu_sublist(scpus, smt=True):
    '''Produce a CPU list with/without SMT pairs - resource leaf case'''
    cpu_list = []
    for value in scpus:
        exists = False
        if not smt:
            for cpu in cpu_list:
                if cpu["core"] == value["core"]:
                    exists = True
                    break
        if not exists:
            cpu_list.append(value)
    mask = []
    for cpu in cpu_list:
        mask.append(cpu["cpu"])

    return {"mask":set(mask), "available": True}


def create_omp_places(resources, strategy, smt=True):
    '''Parse CPU topology and generate possible CPU masks'''
    omp_places = []
    if strategy == "all":
        omp_places.append(produce_cpu_list(resources["cpus"], smt))
    elif strategy == "cores":
        for value in resources["cores"].values():
            omp_places.append(produce_cpu_sublist(value, smt))
    elif strategy == "nodes":
        for value in resources["nodes"].values():
            omp_places.append(produce_cpu_sublist(value, smt))
    else:
        raise NotImplementedError("Unknown strategy")

    return omp_places

class OMPProcess(multiprocessing.Process):
    '''OMP aware process class'''
    resources = None
    omp_places = []

    def __init__(self, *args, **kwargs):

        try:
            strategy = kwargs["omp_strategy"]
            smt = kwargs["omp_smt"]
        except KeyError:
            strategy = "nodes"
            smt = False
        if OMPProcess.resources is None:
            OMPProcess.resources = enumerate_resources()
        if OMPProcess.omp_places is None:
            OMPProcess.omp_places = create_omp_places(OMPProcess.resources, strategy, smt=smt)
        for place in OMPProcess.omp_places:
            if place["available"]:
                self.place = place
                place["available"] = False
                os.environ["OMP_PLACES"] = "{" + ",".join(list(place["mask"])) + "}"
                # pylint: disable=consider-using-f-string
                os.environ["OMP_NUM_THREADS"] = "{}".format(len(place["mask"]))
                os.environ["OMP_PROC_BIND"] = "True"
        super().__init__(self, *args, **kwargs)

    def join(self, *args, **kwargs):
        '''Grab process termination handling and make the OMP place available again'''
        self.place["available"] = True
        super().join(*args, **kwargs)

    def close(self, *args, **kwargs):
        '''Just in case someone is not using join to wait for the process'''
        self.place["available"] = True
        super().close(*args, **kwargs)
