import os
import matplotlib.pyplot as plt
from pathlib import Path


"""
use it only when the PySR has already generated the tex files
usage: python generete-equations-img.py
"""


def tex_to_image(input_dir="./pysr_equations", output_dir="./pysr_equations"):
    # Create output folder
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Configure Matplotlib for clean math rendering
    plt.rcParams.update(
        {
            "text.usetex": False,  # Uses internal mathtext engine (no local LaTeX needed)
            "font.family": "serif",
            "mathtext.fontset": "cm",  # Computer Modern (Classic LaTeX look)
        }
    )

    for tex_file in Path(input_dir).glob("*.tex"):
        with open(tex_file, "r") as f:
            formula = f.read().strip()

        # Ensure formula is wrapped in $ for mathtext
        if not formula.startswith("$"):
            formula = f"${formula}$"

        print(f"Generating image for: {tex_file.name}")

        # Create a figure
        fig = plt.figure(figsize=(max(len(formula) * 0.2, 2), 1))
        fig.text(0.5, 0.5, formula, size=24, va="center", ha="center")

        # Save as PNG (you can change extension to .pdf if preferred)
        output_path = Path(output_dir) / f"{tex_file.stem}.png"
        plt.savefig(
            output_path, bbox_inches="tight", pad_inches=0.1, transparent=False, dpi=300
        )
        plt.close(fig)


if __name__ == "__main__":
    tex_to_image(input_dir="./pysr_equations", output_dir="./pysr_equations")
