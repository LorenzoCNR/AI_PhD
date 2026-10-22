"""Short deterministic CPU qualification, using generated toy TRAIN data only."""
import argparse
import hashlib
import inspect
import json
import platform
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .loss import evaluate_soft,fit_distance_scales

def field(result,key):
    return result[key] if isinstance(result,dict) else getattr(result,key)

def run_toy(version="soft_v1_stable",steps=30,seed=42):
    import cebra.models
    if steps<1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    position=torch.linspace(-1,1,8).repeat(4)
    time_id=torch.arange(8).repeat(4)
    trial_id=torch.arange(4).repeat_interleave(8)
    behavior=torch.stack((position,torch.ones_like(position),torch.zeros_like(position)),1)
    batch=dict(behavior=behavior,time_id=time_id,trial_id=trial_id,session_id=torch.zeros_like(trial_id))
    specs=[dict(key="behavior",geometry="euclidean",weight=.5),dict(key="time_id",geometry="temporal",weight=.5)]
    scales=fit_distance_scales(batch,specs,split="train",source_ids=torch.arange(32))
    signal=torch.stack((position,position.square(),torch.sin(position*3),torch.cos(position*3)),1)
    neural=signal[:,:,None].expand(-1,-1,10).clone()+.03*torch.randn(32,4,10)
    encoder=cebra.models.init("offset10-model",num_neurons=4,num_units=32,num_output=32).cpu()
    optimizer=torch.optim.Adam(encoder.parameters(),lr=1e-3)
    history=[]
    started=time.perf_counter()
    encoder.eval()
    with torch.no_grad():
        initial=field(evaluate_soft(encoder(neural),batch,specs,version=version,scales=scales),"loss").item()
    encoder.train()
    for update in range(steps):
        optimizer.zero_grad(set_to_none=True)
        result=evaluate_soft(encoder(neural),batch,specs,version=version,scales=scales)
        loss=field(result,"loss")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CPU toy loss")
        loss.backward()
        norm_squared=0.
        for parameter in encoder.parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Non-finite CPU toy gradient")
                norm_squared+=parameter.grad.square().sum().item()
        optimizer.step()
        history.append(dict(update=update+1,loss=loss.item(),gradient_norm=norm_squared**.5))
    encoder.eval()
    with torch.no_grad():
        final=evaluate_soft(encoder(neural),batch,specs,version=version,scales=scales)
    final_loss=field(final,"loss").item()
    if not final_loss<initial:
        raise RuntimeError(f"Fixed-support toy loss did not improve: {initial} -> {final_loss}")
    diagnostics=final.diagnostics() if hasattr(final,"diagnostics") else final.get("diagnostics",{})
    model_source=Path(inspect.getfile(type(encoder)))
    return dict(version=version,seed=seed,actual_optimizer_updates=steps,device="cpu",initial_fixed_support_loss=initial,final_fixed_support_loss=final_loss,seconds=time.perf_counter()-started,history=history,diagnostics=diagnostics,train_scales=scales,encoder_source_path=str(model_source),encoder_source_sha256=hashlib.sha256(model_source.read_bytes()).hexdigest(),parameter_sha256=hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in encoder.parameters())).hexdigest(),scope="software qualification only; generated toy TRAIN; no scientific efficacy claim")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=30)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Immutable qualification path exists: {args.output}")
    records=[run_toy(version,args.steps,args.seed) for version in ("soft_v1_frozen","soft_v1_stable")]
    args.output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),torch_version=torch.__version__,python_version=platform.python_version(),seed=args.seed,device="cpu",split="toy_train",parent_data="deterministic generated toy",source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},config=dict(steps=args.steps,seed=args.seed,architecture="official offset10-model",num_units=32,output_dimension=32,learning_rate=1e-3),no_project_test_data_accessed=True)
    (args.output/"CPU_SMOKE_RESULTS.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    (args.output/"PROVENANCE.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output),versions=[{k:r[k] for k in ("version","actual_optimizer_updates","initial_fixed_support_loss","final_fixed_support_loss","seconds")} for r in records]),indent=2))

if __name__=="__main__":
    main()
