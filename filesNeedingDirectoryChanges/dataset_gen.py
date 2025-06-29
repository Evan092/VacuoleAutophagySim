#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import random

def random_params():
    N = random.randint(5, 10)
    mu    = random.uniform(5, 6.5)
    sigma = random.uniform(0.1, 0.5)

    # vacuole phys. radius ~2 000–3 000 nm → vox radius ~40–60 @ dx=50
    wall_mu    = random.uniform(7.6, 8.0)   # exp(7.6)=2000, exp(8.0)=2980
    wall_sigma = random.uniform(0.3, 0.6)

    pvals = 2.0
    mu_body_number    = random.uniform(1.0, 3.0)
    sigma_body_number = random.uniform(0.2, 1.0)

    # pick dx so that vac_diameter ≲136 voxels:
    dx = random.uniform(40.0, 60.0)

    show_wall    = "True"
    optimmaxiter = random.randint(50, 500)
    seed         = random.randint(1, 1_000_000)
    iterations   = random.randint(2, 10)
    piff         = 1

    return (
        N, mu, sigma,
        wall_mu, wall_sigma,
        pvals,
        mu_body_number, sigma_body_number,
        dx, show_wall,
        optimmaxiter,
        seed, iterations,
        piff
    )

def main():
    num_samples = 100
    base_out = os.path.abspath("runs")
    os.makedirs(base_out, exist_ok=True)

    for i in range(1, num_samples + 1):
        (
            N, mu, sigma,
            wall_mu, wall_sigma,
            pvals,
            mu_body_number, sigma_body_number,
            dx, show_wall,
            optimmaxiter,
            seed, iterations,
            piff
        ) = random_params()

        run_folder = os.path.join(base_out, f"run_{i:02d}")
        if os.path.exists(run_folder):
            shutil.rmtree(run_folder)
        os.makedirs(run_folder)

        print("─" * 80)
        print(
            f"[{i}/{num_samples}] "
            f"N={N}, μ={mu:.3f}, σ={sigma:.3f}, "
            f"wall_μ={wall_mu:.3f}, wall_σ={wall_sigma:.3f}, "
            f"p={pvals:.2f}, μ_body={mu_body_number:.2f}, σ_body={sigma_body_number:.2f}, "
            f"dx={dx:.1f}, show_wall={show_wall}, maxiter={optimmaxiter}, "
            f"seed={seed}, iters={iterations}, PIFF={piff}"
        )

        cmd = [
            "python", "vacuole_gen.py",
            "--run_folder", run_folder,
            "--N", str(N),
            "--mu", f"{mu:.3f}",
            "--sigma", f"{sigma:.3f}",
            "--wall_radius_mu", f"{wall_mu:.3f}",
            "--wall_radius_sigma", f"{wall_sigma:.3f}",
            "--pvals", f"{pvals:.3f}",
            "--mu_body_number", f"{mu_body_number:.3f}",
            "--sigma_body_number", f"{sigma_body_number:.3f}",
            "--dx", f"{dx:.3f}",
            "--show_wall", show_wall,
            "--optimmaxiter", str(optimmaxiter),
            "--seed", str(seed),
            "--iterations", str(iterations),
            "--PIFF", str(piff),
        ]

        subprocess.run(cmd, check=True)

        shutil.copyfile('CompuCell3D/cc3dSimulation/Simulation/clustertest.xml', run_folder + "/clustertest.xml")
        shutil.copyfile('output.piff', run_folder + "/output.piff")

if __name__ == "__main__":
    main()
