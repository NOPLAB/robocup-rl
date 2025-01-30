# robocup-rl

## genesis_sb3

Install dependencies 

```bash
pip install 'stable-baselines3[extra]'
pip install gymnasium
```

Run

```bash
git clone https://github.com/NOPLAB/robocup-rl
cd robocup-rl/genesis_sb3

python robocup_train.py
python robocup_eval.py

tensorboard --logdir logs
```
