import os
import shutil
import subprocess

BASE_DIR = r"C:\Users\evans\Desktop\IndependentStudy"
RUNS_DIR = os.path.join(BASE_DIR, "runs")
SIM_DIR = os.path.join(BASE_DIR, "CompuCell3D", "cc3dSimulation", "Simulation")

OUTPUT_DIR = os.path.join(BASE_DIR, "cc3dSimulation", "outputs")
OUTPUT_DIR2 = os.path.join(BASE_DIR, "CompuCell3D", "cc3dSimulation")
SCRIPT_PATH = os.path.join(BASE_DIR, "CompuCell3D", "runScript.bat")
CC3D_PROJECT = os.path.join(BASE_DIR, "CompuCell3D", "cc3dSimulation", "CC3D_sim_for_AVS.cc3d")

for i in range(22, 101):
    run_id = f"run_{i:02d}"
    run_path = os.path.join(RUNS_DIR, run_id)
    
    if not os.path.isdir(run_path):
        continue

    print(f"--- Processing {run_id} ---")

    # Copy clustertest.xml
    xml_src = os.path.join(run_path, "clustertest.xml")
    xml_dst = os.path.join(SIM_DIR, "clustertest.xml")
    if not os.path.isfile(xml_src):
        print(f"Missing clustertest.xml in {run_path}, skipping...")
        continue
    shutil.copy2(xml_src, xml_dst)

    # Copy output.piff
    piff_src = os.path.join(run_path, "output.piff")
    piff_dst = os.path.join(OUTPUT_DIR2, "output.piff")
    if not os.path.isfile(piff_src):
        print(f"Missing output.piff in {run_path}, skipping...")
        continue
    shutil.copy2(piff_src, piff_dst)

    for j in range(1, 6):
        print(f"Running simulation {j}/5 for {run_id}")

        subprocess.run(
            [SCRIPT_PATH, "-i", CC3D_PROJECT, "-f", "1", "-o", OUTPUT_DIR, "-c", "infoPrinter"],
            cwd=BASE_DIR,
            shell=True
        )

        # Rename and copy outputs folder
        output_label = f"outputs_{j:02d}"
        out_target = os.path.join(run_path, output_label)

        if os.path.isdir(OUTPUT_DIR):
            if os.path.exists(out_target):
                shutil.rmtree(out_target)
            shutil.copytree(OUTPUT_DIR, out_target)
            shutil.rmtree(OUTPUT_DIR)
        else:
            print(f"Output folder missing after run {j} of {run_id}")
