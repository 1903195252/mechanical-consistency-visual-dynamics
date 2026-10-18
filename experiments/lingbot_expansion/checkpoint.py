"""Exact training restart: model, optimizer, scheduler, sampler and all RNGs."""
import random
import os
import hashlib
import pickle
from pathlib import Path
import numpy as np
import torch


def fingerprint(model,optimizer,scheduler,order):
    """Streaming digest of every model/Adam tensor, plus schedule and data cursor."""
    digest=hashlib.sha256()
    def visit(value):
        if isinstance(value,torch.Tensor):
            local=value.to_local() if hasattr(value,'to_local') else value
            local=local.detach().cpu().contiguous()
            digest.update(str((local.dtype,tuple(local.shape))).encode())
            digest.update(local.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value,key=str):digest.update(repr(key).encode());visit(value[key])
        elif isinstance(value,(tuple,list)):
            digest.update(str(type(value)).encode())
            for entry in value:visit(entry)
        else:digest.update(pickle.dumps(value,protocol=4))
    for obj in (model,optimizer,scheduler,order):visit(obj.state_dict())
    return digest.hexdigest()


class SequenceOrder:
    """Explicit shuffled sequence order, independent of diffusion-noise RNG."""
    def __init__(self,length,seed):
        if length<1:raise ValueError('Empty training set')
        self.length=length;self.generator=torch.Generator().manual_seed(seed)
        self.order=[];self.cursor=0;self.epochs=0

    def next(self):
        if self.cursor==len(self.order):
            self.order=torch.randperm(self.length,generator=self.generator).tolist()
            self.cursor=0;self.epochs+=1
        value=self.order[self.cursor];self.cursor+=1
        return value

    def state_dict(self):
        return dict(length=self.length,order=self.order,cursor=self.cursor,epochs=self.epochs,generator=self.generator.get_state())

    def load_state_dict(self,state):
        if state['length']!=self.length:raise ValueError('Dataset size changed')
        self.order=state['order'];self.cursor=state['cursor'];self.epochs=state['epochs'];self.generator.set_state(state['generator'])


def save(path,model,optimizer,scheduler,order,step,identity,distributed=False):
    if distributed:
        from torch.distributed.checkpoint.state_dict import get_state_dict,StateDictOptions
        model_state,optimizer_state=get_state_dict(model,optimizer,options=StateDictOptions(full_state_dict=True,cpu_offload=True))
    else:model_state,optimizer_state=model.state_dict(),optimizer.state_dict()
    state=dict(model=model_state,optimizer=optimizer_state,scheduler=scheduler.state_dict(),
        order=order.state_dict(),step=step,identity=identity,rng_python=random.getstate(),
        rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    path=Path(path);partial=path.with_suffix('.partial');path.parent.mkdir(parents=True,exist_ok=True)
    with partial.open('wb') as stream:
        torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
    partial.replace(path)


def restore(path,model,optimizer,scheduler,order,identity,distributed=False):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']!=identity:raise ValueError('Checkpoint dataset/protocol/seed identity changed')
    if distributed:
        from torch.distributed.checkpoint.state_dict import set_state_dict,StateDictOptions
        set_state_dict(model,optimizer,model_state_dict=state['model'],optim_state_dict=state['optimizer'],
                       options=StateDictOptions(full_state_dict=True,cpu_offload=True,strict=True))
    else:
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler']);order.load_state_dict(state['order'])
    random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);torch.set_rng_state(state['rng_torch'])
    if state['rng_cuda'] is not None:torch.cuda.set_rng_state_all(state['rng_cuda'])
    return state['step']
