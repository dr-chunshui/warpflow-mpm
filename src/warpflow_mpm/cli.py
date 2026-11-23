"""Command-line entry point: warpflow run config.yaml"""

import argparse


def main():
    ap = argparse.ArgumentParser(
        prog="warpflow",
        description="warpflow-mpm: GPU MPM solver (Warp + MPI)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run", help="run a YAML config")
    runp.add_argument("config", help="path to YAML config file")
    runp.add_argument("--log", default="warpflow.log.csv",
                      help="energy log CSV path")

    args = ap.parse_args()

    if args.cmd == "run":
        from warpflow_mpm.config import load_config
        from warpflow_mpm.simulation import Simulation

        cfg = load_config(args.config)
        sim = Simulation(cfg, log_path=args.log)
        sim.run()


if __name__ == "__main__":
    main()
