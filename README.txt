EASY V2: 300x300, 2 obstacles, 50x50 target. Observation 64 floats.
1. Back up your envs/generic_env.py and envs/obstacles_env.py.
2. Copy both .py files from envs/ into the repository envs/ folder.
   IMPORTANT: Keep your current envs/obstacle.py (Ray-AABB implementation).
3. Copy patch_dqn.py and test_easy_v2.py to repository root.
4. Run python patch_dqn.py to update observation_space=64 and map 300x300.
5. Run python test_easy_v2.py. Then run python dqn.py.
6. Old 63-input checkpoints are INCOMPATIBLE. Retrain from scratch.
7. Only the observation and reward are revised. Existing timeout/replay logic
   in your dqn.py is NOT rewritten by patch_dqn.py; verify it separately.
8. Change old tests that assert shape==(63,) to assert shape==(64,).
