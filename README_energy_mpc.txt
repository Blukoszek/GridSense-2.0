
Quickstart — Energy MPC (files in this ZIP)

1) Install deps (Python 3.10+):
   pip install pulp pandas matplotlib pytz

   If PuLP can't find a solver, install CBC:
   - macOS:  brew install coin-or-tools/coinor/cbc
   - Debian/Ubuntu: sudo apt-get install -y coinor-cbc

2) Files:
   - energy_mpc.py
   - config_default.json
   - input_template.csv   (headers example)
   - input_demo.csv       (24h synthetic data; run this first)
   - README_energy_mpc.txt

3) Run (example):
   python energy_mpc.py --csv input_demo.csv --config config_default.json --outdir out --horizon-hours 24 --step-min 5 --validate
