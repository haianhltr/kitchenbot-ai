# Configs

A run without a committed config did not happen (AGENTS.md section 6).

```
configs/
├── train/   one YAML per training run family, e.g. stand_g1_dex3.yaml (P05)
└── eval/    one YAML per scenario and split, e.g. s4_unseen.yaml (P23)
```

Every file that names a physics preset must set `physics: isaacsim_physx`; `tests/test_package.py`
fails otherwise. Configs record `num_envs`, seeds, iteration budgets and randomization ranges, and
the evaluator writes the config's hash into every result JSON.
