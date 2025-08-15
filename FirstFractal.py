import torch
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# L-system rules
axiom = "A"
rule1 = "A-B--B+A++AA+B-"
rule2 = "+A-BB--B-A++A+B"
iterations = 8

# Initial coordinates
coordinates = torch.tensor([[0.0, -1.0], [0.0, 0.0]], device=device)

for _ in range(iterations):
    # --- Expand axiom in parallel ---
    substr = []
    for k in axiom:
        if k == "A":
            substr.append(rule1)
        elif k == "B":
            substr.append(rule2)
        else:
            substr.append(k)
    substr = ''.join(substr)

    # --- Convert characters to ASCII tensor for parallel processing ---
    substr_tensor = torch.tensor([ord(c) for c in substr], device=device)

    # --- Parallel angle assignment ---
    angles = torch.zeros(len(substr), device=device)
    angles[substr_tensor == ord('+')] = 60.0
    angles[substr_tensor == ord('-')] = -60.0
    # Others remain 0

    # --- Cumulative rotation in radians ---
    theta = torch.cumsum(angles, dim=0) * (torch.pi / 180)

    # --- Direction mask for A/B symbols ---
    directions = ((substr_tensor == ord('A')) | (substr_tensor == ord('B'))).float()

    # --- Unit vectors for all symbols ---
    unit_vectors = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * directions.unsqueeze(1)

    # --- Compute displacements ---
    last_vector = coordinates[-1] - coordinates[-2]
    displacement_norm = torch.norm(last_vector)
    displacements = unit_vectors * displacement_norm

    # Only keep displacements where directions == 1
    displacements = displacements[directions == 1]

    # --- Append new coordinates ---
    if len(displacements) > 0:
        coordinates = torch.cat([coordinates, torch.cumsum(displacements, dim=0) + coordinates[-1]], dim=0)

    # --- Update axiom for next iteration ---
    axiom = substr

# --- Move to CPU and plot ---
coords_cpu = coordinates.cpu()
plt.plot(coords_cpu[:, 0], coords_cpu[:, 1], c="black", linewidth=0.2)
plt.axis("equal")
plt.show()
