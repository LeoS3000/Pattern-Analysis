import matplotlib.pyplot as plt
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# L-system rules
axiom = "A"
rule1 = "A-B--B+A++AA+B-"
rule2 = "+A-BB--B-A++A+B"

iterations = 4

# Initial coordinates
coordinates = torch.tensor([[0.0, -1.0], [0.0, 0.0]], device=device)

for _ in range(iterations):
    # Expand the axiom
    substr = ""
    for k in axiom:
        if k == "A":
            substr += rule1
        elif k == "B":
            substr += rule2
        else:
            substr += k

    stack = []
    count = 0

    for i in substr:
        if i == "+":
            stack.append(60.0)
        elif i == "-":
            stack.append(-60.0)
        elif i in "AB":
            if count != 0:
                # Sum angles in stack
                angle = torch.tensor(sum(stack), device=device)
                stack = []

                # Convert degrees to radians manually
                theta = angle * (torch.pi / 180)

                # Rotation matrix
                R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                  [torch.sin(theta),  torch.cos(theta)]], device=device)

                # Compute line vector
                line = coordinates[-1] - coordinates[-2]

                # Rotate line
                vector = R @ line

                # Append new coordinate
                new_point = coordinates[-1] + vector
                coordinates = torch.cat([coordinates, new_point.unsqueeze(0)], dim=0)

            count += 1

    axiom = substr

# Move coordinates back to CPU for plotting
coords_cpu = coordinates.cpu()
plt.plot(coords_cpu[:, 0], coords_cpu[:, 1], c="black")
plt.axis("equal")
plt.show()
