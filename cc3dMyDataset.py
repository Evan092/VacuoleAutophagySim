#!/usr/bin/env python3
import os
import shutil
import subprocess

def main():
    # absolute paths for CC3D
    run_script = "/home/evan/AutomatePipeline/src/CompuCell3D/runScript.sh"
    input_cc3d = "/home/evan/AutomatePipeline/src/CompuCell3D/cc3dSimulation/CC3D_sim_for_AVS.cc3d"
    output_dir = "/home/evan/AutomatePipeline/src/CompuCell3D/cc3dSimulation/outputs/"
    sim_dest   = "/home/evan/AutomatePipeline/src/CompuCell3D/cc3dSimulation/Simulation"

    os.makedirs(sim_dest, exist_ok=True)

    runs_dir = os.path.join(os.getcwd(), "runs")
    for run_name in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue

        # copy simulation inputs
        for fname in ("clustertest.xml", "output.piff"):
            src = os.path.join(run_path, fname)
            dst = os.path.join(sim_dest, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)

        # invoke CC3D
        cmd = [
            run_script,
            "-i", input_cc3d,
            "-f", "1",
            "-o", output_dir,
            "-c", "infoPrinter"
        ]
        subprocess.run(cmd, cwd=sim_dest, check=True)
        print(f"? Completed simulation for {run_name}")

if __name__ == "__main__":
    main()
