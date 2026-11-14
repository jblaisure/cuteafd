#!/usr/bin/env python3
"""Reference-only teacher forcing from exact native F32 input arenas.

No state reconstruction: completed blocks are read from the native arena
prefix whose count is recorded by the producer, and partial is its explicit
input dump. One reference layer runs independently, with native boundaries.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python/reference'))
import numpy as np
import torch
from families.inkling2 import model as f32
from families.inkling2 import native_boundary as boundary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--native-dumps',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--layers',default='')
    args=parser.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    cfg=f32.Config.from_dict(json.loads((args.snapshot/'config.json').read_bytes()))
    weights=f32.Weights(args.snapshot,'cuda')
    weights.validate(cfg)
    records=[]
    args.out.mkdir(parents=True,exist_ok=False)
    def read(path,dtype,shape):
        return torch.from_numpy(np.fromfile(path,dtype=dtype).reshape(shape).copy()).cuda()
    def metrics(actual,expected):
        a,b=actual.double().flatten(),expected.double().flatten()
        return dict(rms=float((a-b).norm()/b.norm()),cosine=float((a@b)/(a.norm()*b.norm())),max_abs=float((a-b).abs().max()))
    with torch.inference_mode():
        folders=sorted(args.native_dumps.glob('layer-*'),key=lambda p:int(p.name.split('-')[1]))
        selected={int(x) for x in args.layers.split(',') if x}
        for folder in folders:
            index=int(folder.name.split('-')[1])
            if selected and index not in selected:
                continue
            meta=json.loads((folder/'meta.json').read_bytes())
            rows,h=meta['rows'],meta['hidden']
            partial=read(folder/'input_partial.bin','<f4',(rows,h))
            arena=read(folder/'input_arena.bin','<f4',(meta['arena_candidates'],meta['arena_rows'],h))
            blocks=[arena[c,:rows] for c in range(meta['completed_blocks'])]
            layer=weights.load_layer(index,cfg)
            taps={}
            result=boundary.layer_forward(partial,blocks,layer,cfg,index,taps=taps)
            native=read(folder/'partial.bin','<f4',(rows,h))
            ids=read(folder/'route_ids.bin','<i4',(rows,cfg.num_experts_per_tok))
            refids=taps['route_ids']
            changed=(ids.sort(-1).values!=refids.sort(-1).values).any(-1)
            logits=read(folder/'route_logits.bin','<f4',(rows,cfg.n_routed_experts))
            record=dict(layer=index,local=meta['local'],provenance=meta['provenance'],
                residual=metrics(native,result),router=metrics(logits,taps['route_logits']),
                expert_sets_equal_rows=int((~changed).sum()),rows=rows,
                flip_rows=torch.where(changed)[0].cpu().tolist(),margins=taps['route_margin'][changed].cpu().tolist())
            folder_out=args.out/f'layer-{index}';folder_out.mkdir()
            for name,value in taps.items():
                if value.dtype==torch.bfloat16:
                    value.cpu().view(torch.uint16).numpy().tofile(folder_out/f'{name}.bin')
                else:
                    value.cpu().numpy().tofile(folder_out/f'{name}.bin')
            records.append(record)
            print(json.dumps(record),flush=True)
            (args.out/'teacher-forced.json').write_text(json.dumps(records,indent=2)+'\n')
            del partial,arena,blocks,layer,result,native,ids,refids,logits,taps
            torch.cuda.empty_cache()


if __name__=='__main__':
    main()
