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
        if lscpu["cores"].get(core, None) is None:
            lscpu["cores"][core] = [cpu]
        else:
            lscpu["cores"][core].append(cpu)
        node = int(cpu["node"])
        if lscpu["nodes"].get(node, None) is None:
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
    omp_places = []

    def __init__(self, *args, **kwargs):

        self.place = None
        if kwargs.get("omp_auto_config", None) is None:
            super().__init__(self, *args, **kwargs)

        try:
            strategy = kwargs["omp_strategy"]
            smt = kwargs["omp_smt"]
        except KeyError:
            strategy = "nodes"
            smt = False

        if OMPProcess.omp_places is None:
            resources = enumerate_resources()
            OMPProcess.omp_places = create_omp_places(resources, strategy, smt=smt)

        for place in OMPProcess.omp_places:
            if place["available"]:
                self.place = place
                place["available"] = False
                os.environ["OMP_PLACES"] = "{" + ",".join(list(place["mask"])) + "}"
                # pylint: disable=consider-using-f-string
                if os.environ.get("OMP_NUM_THREADS", None) is None:
                    os.environ["OMP_NUM_THREADS"] = "{}".format(len(place["mask"]))
                os.environ["OMP_PROC_BND"] = "True"

        super().__init__(self, *args, **kwargs)

    def join(self, *args, **kwargs):
        '''Grab process termination handling and make the OMP place available again'''
        if self.place is not None:
            self.place["available"] = True
        super().join(*args, **kwargs)

    def close(self, *args, **kwargs):
        '''Just in case someone is not using join to wait for the process'''
        if self.place is not None:
            self.place["available"] = True
        super().close(*args, **kwargs)


def parse_mask(mask):
    '''Expand a X-Y,Z list'''
    result = []
    for token in mask.split(","):
        try:
            start, finish = token.split("-")
            if start > finish:
                raise IndexError("Invalid Indexes for cpu ranges")
            for cpu in range(int(start), int(finish) + 1):
                result.append(cpu)
        except ValueError:
            result.append(token)
    return set(result)




class OMPVLLMProcess(OMPProcess):
    '''VLLM Specific class doing additional places split on "|" in the
       VLLM_OMP_PROC_BIND environment variable'''
    def __init__(self, *args, **kwargs):

        vllm_mask = os.environ.get("VLLM_CPU_OMP_THREADS_BIND", None)
        if vllm_mask is None or OMPProcess.omp_places is not None:
            super().__init__(self, omp_auto_config=True, *args, **kwargs)
        if vllm_mask == "nobind":
            super().__init__(self, *args, **kwargs)

        for mask in vllm_mask.split("|"):
            resources = enumerate_resources(mask)
            OMPProcess.omp_places.extend(create_omp_places(resources, "cores", smt=False))

        super().__init__(self, *args, omp_auto_config=True, **kwargs)
