"""Native Q-learning and exact frozen-policy cycle evaluation."""
import ctypes
import hashlib
import os
import shlex
import sys
from pathlib import Path
import random
import subprocess
import numpy as np

def compile_q_kernel(build_dir):
    build_dir=Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('q_learning.cpp')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    target=build_dir/f'qlearning_{digest}{suffix}'
    if not target.exists():
        compiler = shlex.split(os.environ.get('CXX', 'clang++' if sys.platform == 'darwin' else 'c++'))
        subprocess.run([*compiler,'-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(target)],check=True)
    lib=ctypes.CDLL(str(target))
    double=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    integer64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.pricing_q_response.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_uint64,
        ctypes.c_double,ctypes.c_double,ctypes.c_double,double,double,double,ctypes.c_int,
        ctypes.c_int,integer,double,ctypes.c_int,ctypes.c_int,double,integer,double,integer64,double,
        ctypes.c_int,integer]
    lib.pricing_q_response.restype=None
    handle = ctypes.c_void_p
    byte = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    lib.pricing_q_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, double,
        ctypes.c_int, ctypes.c_int]
    lib.pricing_q_create.restype = handle
    lib.pricing_q_advance.argtypes = [handle, double, double, byte]
    lib.pricing_q_advance.restype = ctypes.c_int
    lib.pricing_q_export.argtypes = [handle, double, integer, double, integer64]
    lib.pricing_q_export.restype = ctypes.c_int
    lib.pricing_q_destroy.argtypes = [handle]
    lib.pricing_q_destroy.restype = None
    return lib


class QLearningSession:
    """Own a native response that can pause before an unknown market payoff."""

    def __init__(self, kernel, initial_q, seed, steps, *, alpha, delta, beta,
                 stability_required=0):
        self.handle = None
        self.kernel = kernel
        initial_q = np.ascontiguousarray(initial_q, dtype=np.float64)
        self.m = initial_q.shape[-1]
        if initial_q.shape != (2, self.m*self.m, self.m) or steps < 1:
            raise ValueError('A response needs positive steps and two memory-one Q tables')
        rng = random.Random(int(seed))
        self.initial_joint = self.m*rng.randint(0, self.m-1) + rng.randint(0, self.m-1)
        self.handle = kernel.pricing_q_create(self.m, int(steps), int(seed),
            alpha, delta, beta, initial_q, self.initial_joint, int(stability_required))
        if not self.handle:
            raise RuntimeError('Cannot allocate native seller response')

    def advance(self, utility, surplus, known):
        if self.handle is None:
            raise RuntimeError('Seller response is closed')
        if (np.shape(utility) != (self.m, self.m, 2)
                or np.shape(surplus) != (self.m, self.m)
                or np.shape(known) != (self.m, self.m)):
            raise ValueError('Payoffs and known-entry mask must match the price grid')
        pair = self.kernel.pricing_q_advance(self.handle,
            np.ascontiguousarray(utility, dtype=np.float64),
            np.ascontiguousarray(surplus, dtype=np.float64),
            np.ascontiguousarray(known, dtype=np.uint8))
        return None if pair < 0 else divmod(pair, self.m)

    def result(self):
        if self.handle is None:
            raise RuntimeError('Seller response is closed')
        q = np.empty((1, 2, self.m*self.m, self.m), dtype=np.float64)
        visits = np.empty(q.shape, dtype=np.int64)
        joint = np.empty(1, dtype=np.int32)
        stats = np.empty((1, 12), dtype=np.float64)
        rounds = self.kernel.pricing_q_export(self.handle, q, joint, stats, visits)
        assert np.isfinite(q).all() and np.isfinite(stats).all()
        assert np.all(visits.sum(axis=(2, 3)) == rounds)
        return dict(q=q, visits=visits, joint=joint, stats=stats,
                    actual_steps=rounds, initial_joint=self.initial_joint)

    def close(self):
        if self.handle is not None:
            self.kernel.pricing_q_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def __del__(self):
        self.close()

def learn(kernel,U,W,Q,seed,checkpoints,memory,alpha=.25,delta=.95,beta=1e-4,
          initial_joint=None,tape=None,trace_length=0,stability_required=0):
    """Execute stateful or stateless Q-learning with uninterrupted checkpoints."""
    checkpoints=np.ascontiguousarray(checkpoints,dtype=np.int32)
    assert len(checkpoints)>0 and checkpoints[0]>0 and np.all(np.diff(checkpoints)>0)
    m=np.asarray(W).shape[0]
    assert np.asarray(W).shape==(m,m) and np.asarray(U).shape==(m,m,2)
    states=m*m if memory else 1
    Q=np.ascontiguousarray(Q,dtype=float)
    assert Q.shape==(2,states,m)
    if initial_joint is None:
        rng=random.Random(seed)
        initial_joint=m*rng.randint(0,m-1)+rng.randint(0,m-1)
    assert 0<=initial_joint<m*m
    saved=np.zeros((len(checkpoints),2,states,m))
    visits=np.zeros(saved.shape,dtype=np.int64)
    joints=np.zeros(len(checkpoints),dtype=np.int32)
    stats=np.zeros((len(checkpoints),12)); trace=np.zeros((max(1,trace_length),8))
    actual_steps=np.zeros(1,dtype=np.int32)
    if tape is not None:
        tape=np.ascontiguousarray(tape,dtype=float)
        assert tape.shape==(int(checkpoints[-1]),4) and np.all((tape>=0)&(tape<1))
    random_tape=np.zeros(1) if tape is None else tape
    kernel.pricing_q_response(m,int(memory),int(checkpoints[-1]),int(seed),alpha,delta,beta,
        np.ascontiguousarray(U,dtype=float),np.ascontiguousarray(W,dtype=float),Q,int(initial_joint),
        len(checkpoints),checkpoints,random_tape,int(tape is not None),trace_length,saved,joints,stats,visits,trace,
        int(stability_required),actual_steps)
    assert np.isfinite(saved).all() and np.isfinite(stats).all()
    effective=np.minimum(checkpoints,int(actual_steps[0]))
    assert np.array_equal(visits.sum(axis=(2,3)),np.repeat(effective[:,None],2,axis=1))
    return {'q':saved,'joint':joints,'stats':stats,'visits':visits,'trace':trace[:trace_length],
            'initial_joint':int(initial_joint),'actual_steps':int(actual_steps[0])}

def frozen_policy(Q):
    """Freeze first-argmax policies, expanding stateless responses if needed."""
    greedy=np.argmax(Q,axis=2)
    return np.repeat(greedy,Q.shape[2]**2,axis=1) if greedy.shape[1]==1 else greedy

def cycle(policy,start):
    """Return transient and recurrent joint states of a frozen automaton."""
    m=int(np.sqrt(policy.shape[1]));assert m*m==policy.shape[1]
    seen={};path=[];s=int(start)
    while s not in seen:
        seen[s]=len(path);path.append(s)
        s=int(policy[0,s]*m+policy[1,s])
    return path[:seen[s]],path[seen[s]:]
