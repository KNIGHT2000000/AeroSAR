# AeroSAR-Optimizer
### Energy-Aware Autonomous UAV Search-and-Rescue Route Optimization via Asymmetric Bitmask Dynamic Programming and Kinematic Simulation

The complete codebase, documentation, tests, and configuration are organized in the [`aerosar_optimizer/`](file:///d:/DAA_Project/aerosar_optimizer/) directory.

To run the complete pipeline:
```bash
python main.py
```

To run the full benchmark suite across all target counts:
```bash
python main.py --full-benchmark
```

To run the test suite:
```bash
python -m pytest aerosar_optimizer/tests/test_aerosar.py -v
```

To launch the interactive Streamlit GCS Dashboard:
```bash
streamlit run aerosar_optimizer/visualization/dashboard.py
```

See the full documentation and mathematical proofs in [aerosar_optimizer/README.md](file:///d:/DAA_Project/aerosar_optimizer/README.md).
