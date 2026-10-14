# AI_PhD — reproduction guide

This repository contains the project-specific code and protocol for reproducing selected rat and monkey experiments from the CEBRA paper. It is organized around runnable Python scripts, not notebooks; there are currently no project `.ipynb` notebooks.

## What belongs in Git

- `cebra_replica/`: project runners, embedding export, scoring, plotting, and the rat/monkey stage guides.
- `RAT_REPLICATION_PLAN.md`: the rat protocol and comparison decisions.
- `README.md`: this user and maintainer guide.

The large or separately maintained material stays local and is intentionally ignored by Git: upstream checkouts (`CEBRA/`, `SoftCLT/`, `CPC_audio_reference/`), raw datasets (`data/`), all generated experiment outputs (`results/`), and the local reference-library folder (`Ref Bib/`). Ignoring these paths does not delete or move them. The `results/` folder is deliberately not being selected for publication yet; we can curate it separately later.

## Start here: map an action to code

| If you want to… | Start with… | Important behavior |
|---|---|---|
| Read the frozen rat replication choices | `RAT_REPLICATION_PLAN.md` | Protocol, splits, conditions, and paper-comparison caveats. |
| Check the rat campaign without fitting | `python cebra_replica/rat_campaign.py` | Read-only verification/status; it expects local `results/` artifacts. |
| Fit one rat configuration | `python cebra_replica/rat_benchmark.py --help` | A fit writes checkpoint, embeddings, and provenance under ignored `results/`. |
| Launch missing rat campaign fits | `python cebra_replica/rat_campaign.py --execute` | Expensive: this can launch many fits. Review the plan and `--help` first. |
| Score a frozen rat fit | `cebra_replica/rat_score.py` | Takes a run directory; scoring is downstream of the saved embedding. |
| Verify monkey source bundles and split mapping | `python cebra_replica/monkey/prepare_figure3.py` | No model fitting; checks source MD5s and writes local stage-00 provenance/figure. |
| Run the monkey Figure 3 paper-settings fits | `python cebra_replica/monkey/fit_figure3_paper.py --help` | Fit entry point; output goes under ignored `results/monkey_figure3_paper_v1/`. |
| Run the separate monkey held-out Figure 3h fits | `python cebra_replica/monkey/fit_figure3h.py --help` | Separate fit branch and output root. |
| Follow all rat code groups | `cebra_replica/README.md` | Script map, scoring, manifolds, figures, and provenance notes. |
| Follow monkey stages and paper-panel boundaries | `cebra_replica/monkey/README.md` | Data preparation, fits, embeddings, figures, and Extended Data 8. |

Before any fit, use the script's `--help` and confirm the intended output path. Fit scripts do not publish results: they write locally to `results/`, which remains ignored here. Existing local outputs are not removed by this Git configuration.

## Recovering the data

The local `data/` bundle is about 700 MB, so it is not stored in Git. The rat and monkey data have different recovery behavior; neither should be described simply as generated output.

### Rat hippocampus

Expected files are `data/rat_hippocampus/{achilles,buddy,cicero,gatsby}.jl`. The rat runner constructs CEBRA's `SingleRatTrialSplitDataset` with `root=AI_PhD/data`; the upstream loader has `download=True` and downloads/checks the source files when absent. The official URLs and expected checksums are defined in the pinned CEBRA source at `CEBRA/cebra/datasets/hippocampus.py`.

### Monkey Area 2

The monkey preparation script expects twelve `.jl` files in `data/monkey_reaching_preload_smth_40/`: `all`, `active`, and `passive`, each with suffix `all`, `train`, `valid`, and `test`. The official compressed files are served from:

`https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/<name>.jl.gz`

From the repository root, PowerShell can download and decompress the exact twelve bundles:

```powershell
$base = 'https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40'
$dest = 'data/monkey_reaching_preload_smth_40'
New-Item -ItemType Directory -Force $dest | Out-Null
$names = @('all_all','all_train','all_valid','all_test', 'active_all','active_train','active_valid','active_test', 'passive_all','passive_train','passive_valid','passive_test')
foreach ($name in $names) {
    $gz = Join-Path $dest "$name.jl.gz"
    Invoke-WebRequest -Uri "$base/$name.jl.gz" -OutFile $gz
    python -c "import gzip,pathlib,sys; p=pathlib.Path(sys.argv[1]); p.with_suffix('').write_bytes(gzip.decompress(p.read_bytes()))" $gz
}
```

Then verify the files and split mapping:

```powershell
python cebra_replica/monkey/prepare_figure3.py
```

That script does not download the bundles or fit a model. It verifies all twelve published MD5 checksums, reconstructs trial/split correspondence from exact trial contents, and records source hashes. The expected checksums are stored in `cebra_replica/monkey/prepare_figure3.py`; the matching upstream URL/checksum table is in `CEBRA/cebra/datasets/monkey_reaching.py`.

## Reference code and software

The reference repositories are separate local checkouts, not copies maintained by this project. For the CEBRA implementation used by the current scripts, the local reference checkout was at commit `b7e4b9ef6f83e09894beda6fbf585605c94685b3` from `https://github.com/AdaptiveMotorControlLab/CEBRA`. Recreate that checkout locally with:

```powershell
git clone https://github.com/AdaptiveMotorControlLab/CEBRA.git CEBRA
git -C CEBRA checkout b7e4b9ef6f83e09894beda6fbf585605c94685b3
python -m pip install -e .\CEBRA
```

The PiVAE runners also expect the CEBRA checkout's `third_party/pivae` source tree and TensorFlow. `SoftCLT/` and `CPC_audio_reference/` are optional literature/code references, not dependencies of the current rat/monkey replication. This repository does not yet contain a project-wide environment lock file; exact Python/package environment pinning remains to be added before claiming a clean-machine, bit-for-bit reproduction.

## Provenance and outputs

Fit and downstream scripts write versioned artifacts, including checkpoints/embeddings where applicable, run manifests, and hashes, under `results/`. These artifacts remain available in the local working tree but are excluded from Git for now. The historical pre-reorganization rat runner snapshots under `cebra_replica/source_snapshots/pre_layout_2026-10-12/` are retained because the campaign verifier uses their source hash when validating older completed fits.
