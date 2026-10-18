"""Use the already frozen prompt embeddings without loading an unused T5."""
import argparse
from .common import OUT,SEEDS,STEPS,sha,write
from . import run_model


def run(seed,step,shard=0,shards=1):
    import torch
    native=run_model.configure(seed);native.setup_native()
    import wan_va_server
    # run_model.infer replaces encode_prompt with the frozen embeddings before
    # _reset or any video chunk. These unused objects need no model weights.
    wan_va_server.load_text_encoder=lambda *args,**kwargs:torch.nn.Identity()
    wan_va_server.load_tokenizer=lambda *args,**kwargs:None
    folder=OUT/'runs'/('seed%d'%seed)/('step%04d'%step)
    write(folder/('prompt_cache_shard%02d.json'%shard),dict(
        execution_adapter_sha256=sha(__file__),
        frozen_embeddings_sha256=sha(OUT/'shared/text_embeddings.pt'),
        unused_text_encoder_loaded=False,native_denoising_unchanged=True))
    run_model.infer(seed,step,shard,shards)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--step',type=int,choices=STEPS,required=True)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    a=parser.parse_args();run(a.seed,a.step,a.shard,a.shards)
