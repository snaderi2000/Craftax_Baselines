<p align="center">
 <img width="80%" src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax_Baselines/main/images/logo.png" />
</p>

# Craftax Baselines

This repository contains the code for running the baselines from the [Craftax paper](https://arxiv.org/abs/2402.16801).
For packaging reasons, this is separate to the [main repository](https://github.com/MichaelTMatthews/Craftax/).

# Installation
```commandline
git clone https://github.com/MichaelTMatthews/Craftax_Baselines.git
cd Craftax_Baselines
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

## V100 Installation
For V100 machines, use the separate pinned requirements file:

```commandline
conda create -n craftax-v100 python=3.10 -y
conda activate craftax-v100
pip install -U pip

pip install -r requirements_v100.txt \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify the install:

```commandline
python - <<'PY'
import jax, numpy as np, scipy
print("jax", jax.__version__)
print("numpy", np.__version__)
print("scipy", scipy.__version__)
print(jax.devices())
print(jax.lib.xla_bridge.get_backend().platform_version)
PY
```

# Run Experiments

### PPO
```commandline
python ppo.py
```

### PPO-RNN
```commandline
python ppo_rnn.py
```

### ICM
```commandline
python ppo.py --train_icm
```

### E3B
```commandline
python ppo.py --train_icm --use_e3b --icm_reward_coeff 0
```

### RND
```commandline
python ppo_rnd.py
```

# Visualisation
You can save trained policies with the `--save_policy` flag.  These can then be viewed with the `view_ppo_agent` script (pass in the path up to the `files` directory).
