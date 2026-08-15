# Formula-to-code mapping

This document records how the proposed fuzzy reasoning dynamical system is
implemented. It is an implementation note, not an additional theoretical
claim.

## Runtime dependencies

`requirements.txt` is the single dependency specification for the core
pipeline, Hugging Face extraction, Tuned Lens, and imported analysis utilities.
The supported full-experiment runtime is Python 3.10-3.12 on Linux x86_64.
Dependencies are installed on the internet-connected cluster host node into an
environment shared with GPU nodes. `scripts/check_environment.py` verifies
imports on the host and CUDA availability on a GPU node before a real-model
run. Complete Hugging Face model and Tuned Lens directories may be downloaded
on another machine and uploaded under `data/models/` and `data/tuned_lens/`.
vLLM remains an explicitly optional, CUDA-specific backend; the Hugging Face
backend does not import it unless selected.

## Extracted variables

For a model with `L` Transformer blocks, extraction uses the last non-padding
prompt token and stores:

| Paper variable | Cache tensor | Shape | Implementation |
|---|---|---:|---|
| `h_l` | `hidden` | `[N,L+1,D]` | input of block 0 followed by every block output |
| `o_l^att` | `attention` | `[N,L,D]` | output hook on each attention submodule |
| `o_l^MLP` | `mlp` | `[N,L,D]` | output hook on each MLP submodule |
| `P_b(log p_l)` | `belief` | `[N,L+1,R]` | fixed Gaussian projection of logit-lens log probabilities |
| `u_l` | `uncertainty` | `[N,L+1,1]` | categorical entropy of the layer-wise distribution |
| `gamma_l` | `margin` | `[N,L+1,1]` | top-1 minus top-2 probability |

The Gaussian belief projection is fixed by a stored seed. This avoids writing
an `N x (L+1) x vocabulary_size` tensor to disk. It should be described as a
tractable compression for `P_b`. The extraction CLI supports both a standard
logit lens and an optional pretrained Tuned Lens; the selected lens and resource
are stored in the cache metadata.

When TwoHopFact/SOCRATES labels are present, extraction also stores the
first-token log probability of the bridge entity and answer. The bridge signal
supervises a scalar probe on `c_l` and strengthens the composition/transition
semantic prior.

## Reasoning state and operations

`ReasoningProjectors` implements trainable `P_h`, `P_c`, `P_b`, `P_a`, and
`P_m`. The state is constructed as:

```text
s_l = concat(z_l, c_l, b_l, u_l / log(|V|))
```

Attention and MLP operation features are kept outside the state and passed to
the local systems as `a_l` and `m_l`.

## Five local systems

`LocalReasoningDynamics` uses separate parameters and inputs:

| Mode | Network input |
|---|---|
| Knowledge enrichment | `[z_l, m_l]` |
| Information routing | `[z_l, a_l]` |
| Concept composition | `[z_l, c_l, a_l, m_l, a_l * m_l]` |
| Prediction refinement | `[z_l, b_l, u_l]` |
| Hop transition | `[z_l, c_l, delta_c_l, a_l]` |

Each network returns a vector in the full reasoning-state space. Their fuzzy
mixture predicts the observed transition:

```text
predicted_delta_s_l = sum_k mu_lk F_k(...)
```

## Membership dynamics

`MembershipDynamics` implements a recurrent softmax transition using the
previous membership, current state, attention, and MLP features. Its learned
transition matrix is initialized to identity. A learned pre-layer prior is used
to compute the first observed membership.

## Objectives

`fuzzy_dynamics_loss` combines:

- state-transition MSE;
- `KL(reasoning_prior || membership)` semantic regularization;
- squared pairwise cosine similarity between local dynamics;
- negative membership entropy, which encourages fuzzy assignments;
- KL between aggregate mode usage and the uniform distribution;
- bridge-concept probe loss when bridge labels are available.

The proposal writes `-sum(mu log mu)` as an entropy regularizer while stating
that it should prevent hard assignment. Adding positive entropy to a minimized
objective would do the opposite. The implementation therefore minimizes
`sum(mu log mu)`, i.e. negative entropy.

## Paper experiment checklist

Recommended claims should be supported separately by:

1. transition reconstruction on held-out queries/categories;
2. mode diversity and usage statistics;
3. semantic alignment with bridge, answer, attention-flow, and MLP signals;
4. Patchscopes or activation-patching causal interventions;
5. ablations removing semantic, diversity, entropy, and concept-probe losses;
6. comparisons against one global dynamics network, a hard-switching model,
   and an unstructured mixture-of-experts baseline;
7. robustness across models, datasets, random seeds, and projection sizes.

Low reconstruction error alone is insufficient evidence that a mode has the
claimed reasoning interpretation.

## Implemented evaluation metrics

`src/evaluation/metrics.py` implements the metrics proposed in
`ideas/evaluation metrics.md`:

- overall one-step Dynamics R-squared using the mean state-change vector as the
  baseline, rather than an incorrect all-coordinate scalar mean;
- separate `z`, `concept`, `belief`, and normalized-uncertainty reconstruction
  scores and their unweighted Macro-R-squared;
- tie-aware AUROC and average precision for layer-transition events, with
  explicit unavailable results for missing or single-class labels;
- multi-step rollout squared-L2 error, coordinate MSE, final-state error, and
  per-horizon errors.

An R-squared value is undefined when its target block has zero variance. The
JSON result uses `null` and reports target variance/SST rather than silently
forcing that value to zero or one. Strict Macro-R-squared is also `null` if any
of the four blocks is undefined; an explicitly named available-block average is
included for diagnosis.

The rollout recursively uses predicted states, memberships, and concept
changes, but conditions every step on attention/MLP operations extracted from
the observed LLM trajectory. It must therefore be described as an
operation-conditioned rollout, not an autonomous rollout.

Automatic semantic events implement the document's first-token bridge
emergence (F3), joint answer increase/uncertainty decrease (F4), and joint
bridge decrease/answer increase (F5) definitions. These signals overlap with
the semantic training prior. They are diagnostic proxy agreement, not
independent semantic validation. By default, a positive joint signal must also
fall in that query's top quartile; `--semantic-event-quantile` records and
controls this operational definition. The evaluator accepts independent JSON/JSONL
event labels aligned by stable query `uid`; only defined independent events for
all five modes produce the strict Semantic Alignment Macro-AUROC/AP.

## Logging and interruption recovery

Every experiment entry point mirrors progress to a timestamped file under
`logs/`. Core fuzzy-dynamics scripts use structured timestamp/level messages;
the imported latent-multi-hop entry points tee their existing print and tqdm
output into the same directory. During training, `last.pt` and `history.json`
are updated after every epoch, `best.pt` is updated on validation improvement,
and `split.json` is written before the first epoch.

## Time-based experiment layout

Training assigns every invocation a microsecond-resolution `experiment_time`
without asking the user to name it. The timestamp is sortable and prevents
accidental overwrite. The default layout is:

```text
results/fuzzy_dynamics_<experiment_time>/checkpoint/{best.pt,last.pt}
results/fuzzy_dynamics_<experiment_time>/training/{history.json,split.json,experiment_manifest.json,pipeline_manifest.json}
results/fuzzy_dynamics_<experiment_time>/evaluation/{metrics.json,...}
logs/fuzzy_dynamics_<experiment_time>/*.log
```

The checkpoint records `experiment_time`, the source activation cache, and the exact
`split_file`. The evaluator can therefore reconstruct the matching validation
evaluation from `--experiment-time` alone plus the activation cache. Explicit path
overrides remain available for custom or legacy layouts.

`scripts/run_pipeline.py` is the canonical single-experiment orchestrator. It
atomically reserves the experiment directory, passes the same explicit timestamp
to training and evaluation, validates each stage's artifacts, and stops on the
first failed or interrupted stage. It starts from an existing read-only activation
cache; extraction remains a separate one-time step so seed and ablation runs use
identical LLM features. `cluster/run_single_experiment.sbatch` schedules one such
pipeline, while `cluster/run_experiment_array.sbatch` schedules a Cartesian product
of configurations and seeds.
