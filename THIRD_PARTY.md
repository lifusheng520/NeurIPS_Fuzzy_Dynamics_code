# Third-party components

## latent-multi-hop-reasoning

The files under `src/llm/latent_multi_hop`, `scripts/latent_multi_hop`, and
`data/datasets/latent_multi_hop` were adapted or copied from DeepMind's
`latent-multi-hop-reasoning` repository.

- Software license: Apache License 2.0
- Dataset/material license: CC BY 4.0, as documented by the upstream project
- Local license copy: `third_party_licenses/latent-multi-hop-reasoning-Apache-2.0.txt`
- Relevant papers: Yang et al., ACL 2024; Yang et al., arXiv 2024

Local changes are limited to package imports and the corrected Hugging Face
Llama/Qwen layer path (`model.model.layers`).

## FuzzyControl

The files under `src/fuzzy_dynamics/fuzzy_control` are the project author's own
code, copied from `code_from_others/FuzzyControl-main` for subsequent refactoring.
