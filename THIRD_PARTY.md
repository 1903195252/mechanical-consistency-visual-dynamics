# Third-party sources and notices

The model repositories are fetched separately. Their commit identifiers are pinned in `configs/upstream.json`; no pretrained model weights are bundled.

| Component | Source | Terms |
|---|---|---|
| KeyCLD | [rdaems/keycld](https://github.com/rdaems/keycld) | Apache-2.0 |
| CON | [TU Delft coupled oscillator networks](https://github.com/tud-phi/uncovering-iss-coupled-oscillator-networks-from-pixels) | MIT; [upstream notice](LICENSES/CON-MIT.txt) |
| V-JEPA | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812) | MIT; three upstream data-augmentation utilities use Apache-2.0. [MIT notice](LICENSES/VJEPA-MIT.txt) |
| LingBot-VA | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va), [official base model](https://huggingface.co/robbyant/lingbot-va-base) | Apache-2.0 source and base-model distribution. [License text](LICENSES/Apache-2.0.txt) |
| Mechanism geometry reference | `experiments/mechanism_corpus/licenses/mechanism_MIT.txt` | MIT notice retained |
| Mapbox Earcut and bindings | `experiments/mechanism_corpus/licenses/` | ISC notices retained |
| Paper template | `manuscript/keycld_cvpr/cvpr.sty` and bibliography style | Original file headers retained |

`scripts/setup_upstream.py` prepares two import adapters. KeyCLD's unused Torch DataLoader import becomes optional. CON package initializers import only the selected `ConIaeOde`/VAE pathway; unused modules named `con.py` are omitted from this runtime export for filename portability. Selected model and loss source files retain their original bytes. These changes do not alter the model equations.

Experiment adapters provide public geometry, camera and control interfaces. The LingBot adapter replaces the upstream dataset loader, uses native Torch/Flex attention, and streams native BF16 checkpoints while retaining full-parameter training. These adaptations are part of the reported protocol.

Original project code and its documentation are released under [MIT](LICENSE).
Original datasets, rendered observations, physical trajectories, annotations,
saved predictions and numerical results use [CC BY 4.0](DATA_LICENSE.md).
Paper text and publisher/template assets are separate from these grants.

## Artifact license scope

The Zenodo records are mixed-license collections. Licenses apply to the
corresponding components, rather than as alternative licenses for every file.

| Artifact | Applicable terms |
|---|---|
| Original data, observations, trajectories, annotations, predictions and numerical evidence, including those inside model packages | CC BY 4.0 |
| Original experiment, evaluation and import code | MIT |
| KeyCLD model components and checkpoints | Apache-2.0; retain KeyCLD attribution and notices |
| CON model components and checkpoints | MIT; retain the upstream CON notice |
| V-JEPA encoder, AC model components, checkpoints and continuation states | MIT, with upstream Apache-2.0 exceptions retained where applicable |
| LingBot base Transformer, VAE/text components and adapted Transformer checkpoints | Apache-2.0; retain Robbyant and bundled dependency notices |
| Original standalone measurement heads | MIT |

The LingBot adapted checkpoints change the official base Transformer through
the full-parameter, 50-update training protocol documented in the paper.
V-JEPA AC predictors and measurement heads are trained using the documented
experiment adapters; the official V-JEPA encoder is frozen.

Copies of upstream notices are retained in [LICENSES](LICENSES/).
If an individual upstream file or model distribution supplies more specific
terms, those terms remain applicable. The data license does not relicense
third-party models.
