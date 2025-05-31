#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import random

def random_params():
    N = random.randint(5, 30)
    mu = random.uniform(1.6, 2.7)
    sigma = random.uniform(0.1, 0.6)
    wall_mu = random.uniform(3.9, 4.6)
    wall_sigma = random.uniform(0.1, 0.4)
    return N, mu, sigma, wall_mu, wall_sigma

def main():
    num_samples = 5

    for i in range(1, num_samples + 1):
        N, mu, sigma, wall_mu, wall_sigma = random_params()
        
        print("---------------------------------------------------------------------------------------")                
        
        
        print(f"[{i}/{num_samples}] N={N}, mu={mu:.3f}, sigma={sigma:.3f}, wall_mu={wall_mu:.3f}, wall_sigma={wall_sigma:.3f}")

        cmd = [
            "python", "vacuole_gen.py",
            "--N", str(N),
            "--mu", f"{mu:.3f}",
            "--sigma", f"{sigma:.3f}",
            "--wall_radius_mu", f"{wall_mu:.3f}",
            "--wall_radius_sigma", f"{wall_sigma:.3f}",
        ]
        subprocess.run(cmd, check=True)



if __name__ == "__main__":
    main()
