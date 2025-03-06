Run training
```bash
python3 legged_gym/scripts/train.py --task=go2 --num_envs 2048 --experiment_name=test_go2_1 --headless --max_iterations 3000
```

Run evaluation
```bash
python3 legged_gym/scripts/play.py --task=go2 --num_envs 100 --experiment_name=test_go2_1
```