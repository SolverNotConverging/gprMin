"""Generate the exact point orbits used in the native-symmetry documentation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = np.array([[0, -1], [1, 0]])
U = np.diag([-1, 1])
V = np.diag([1, -1])
D = np.array([[0, 1], [1, 0]])
GROUPS = {
    'c4v': ('C4v', [R, U], ['u', 'v', 'd', 'a'], '90° rotations'),
    'c4': ('C4', [R], [], '90° rotations'),
    'c2v': ('C2v · axis mirrors', [U, V], ['u', 'v'], '180° rotation'),
    'c2v_diagonal': ('C2v · diagonal mirrors', [D, -D], ['d', 'a'], '180° rotation'),
    'c2': ('C2', [-np.eye(2, dtype=int)], [], '180° rotation'),
    'cs_u': ('Cs · vertical mirror', [U], ['u'], 'u → −u'),
    'cs_v': ('Cs · horizontal mirror', [V], ['v'], 'v → −v'),
    'cs_diagonal': ('Cs · diagonal mirror', [D], ['d'], '(u, v) → (v, u)'),
    'cs_antidiagonal': ('Cs · antidiagonal mirror', [-D], ['a'], '(u, v) → (−v, −u)'),
    'none': ('C1 · identity only', [], [], 'No nontrivial symmetry'),
}


def render(output=None):
    output = Path(output or Path(__file__).resolve().parents[3] / 'images_shared/symmetry_groups')
    output.mkdir(parents=True, exist_ok=True)
    for key, (title, generators, mirrors, label) in GROUPS.items():
        matrices = [np.eye(2, dtype=int)]
        for matrix in matrices:
            for generator in generators:
                candidate = generator @ matrix
                if not any(np.array_equal(candidate, old) for old in matrices):
                    matrices.append(candidate)
        fig, ax = plt.subplots(figsize=(3.5, 3.5), layout='constrained')
        for point, color in [(np.array([.67, .23]), '#176a9b'), (np.array([.18, .48]), '#d06b20')]:
            orbit = np.array([matrix @ point for matrix in matrices])
            ax.scatter(*orbit.T, color=color, s=50, zorder=3)
        for mirror in mirrors:
            x, y = {'u': ([0, 0], [-1, 1]), 'v': ([-1, 1], [0, 0]),
                    'd': ([-1, 1], [-1, 1]), 'a': ([-1, 1], [1, -1])}[mirror]
            ax.plot(x, y, '--', color='#8e55a0', lw=1.4)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect='equal', xlabel='u', ylabel='v', title=title)
        ax.set_xticks(np.linspace(-1, 1, 9), labels=[])
        ax.set_yticks(np.linspace(-1, 1, 9), labels=[])
        ax.grid(alpha=.18)
        ax.text(.5, .02, label, transform=ax.transAxes, ha='center', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
        destination = output / f'{key}.svg'
        fig.savefig(destination)
        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines()) + '\n')
        plt.close(fig)


if __name__ == '__main__':
    render()
