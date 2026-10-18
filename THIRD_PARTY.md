# Third-party sources and notices

The model repositories are fetched separately. Their commit identifiers are pinned in `configs/upstream.json`; no pretrained model weights are bundled.

| Component | Source | Terms |
|---|---|---|
| KeyCLD | [rdaems/keycld](https://github.com/rdaems/keycld) | Apache-2.0 |
| CON | [TU Delft coupled oscillator networks](https://github.com/tud-phi/uncovering-iss-coupled-oscillator-networks-from-pixels) | MIT |
| V-JEPA | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | Upstream MIT and Apache notices; model weights have their own terms |
| LingBot-VA | [Robbyant/lingbot-va](https://github.com/Robbyant/lingbot-va) | Apache-2.0 source; consult the base-model distribution for weight terms |
| Mechanism geometry reference | `experiments/mechanism_corpus/licenses/mechanism_MIT.txt` | MIT notice retained |
| Mapbox Earcut and bindings | `experiments/mechanism_corpus/licenses/` | ISC notices retained |
| Paper template | `manuscript/keycld_cvpr/cvpr.sty` and bibliography style | Original file headers retained |

`scripts/setup_upstream.py` prepares two import adapters. KeyCLD's unused Torch DataLoader import becomes optional. CON package initializers import only the selected `ConIaeOde`/VAE pathway; unused modules named `con.py` are omitted from this runtime export for filename portability. Selected model and loss source files retain their original bytes. These changes do not alter the model equations.

Experiment adapters provide public geometry, camera and control interfaces. The LingBot adapter replaces the upstream dataset loader, uses native Torch/Flex attention, and streams native BF16 checkpoints while retaining full-parameter training. These adaptations are part of the reported protocol.

The original project code has no assigned redistribution license in this release. Third-party files retain the terms shown in their own notices. Paper text, datasets and pretrained weights are distinct artifacts with their respective terms.
