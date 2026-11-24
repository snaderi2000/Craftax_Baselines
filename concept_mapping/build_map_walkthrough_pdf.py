"""Build an illustrated explanation from explain_map_updates.py artifacts."""
import argparse
import csv
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault('MPLCONFIGDIR', '/tmp/craftax-map-walkthrough-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

INK = '#172635'
BLUE = '#1269ac'
GRAY = '#526578'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args()
    root = args.trace.resolve()
    metadata = json.loads((root/'metadata.json').read_text())
    rows = list(csv.DictReader((root/'trajectory.csv').open()))
    data = np.load(root/'map_snapshots.npz')
    if len(rows) != 16:
        raise ValueError('This report uses the first 15 actions plus the initial state.')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'text.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    output = root/'map_memory_explained.pdf'
    pdf = PdfPages(output)
    pdf.infodict().update(Title='How our Craftax terrain memory works',
                         Subject='A verified 15-step frozen-policy example with explicit odometry',
                         Author='Craftax experiment notes')
    figures = []

    def page(number, title, subtitle):
        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        fig.text(.055, .943, title, fontsize=23, fontweight='bold', color=INK)
        fig.text(.055, .907, subtitle, fontsize=11.5, color=GRAY)
        fig.add_artist(plt.Line2D([.055, .945], [.885, .885], color='#d8e3ed', lw=1.2))
        fig.text(.055, .027, f'Real frozen-policy rollout | seed {metadata["seed"]}, episode {metadata["episode"]} | checkpoint {metadata["checkpoint"]:,}', fontsize=8, color=GRAY)
        fig.text(.945, .027, f'{number} / 5', fontsize=9, ha='right', color=GRAY)
        figures.append(fig)
        return fig

    def paragraph(fig, x, y, text, width=113, size=11.5, color=INK):
        wrapped = '\n'.join(textwrap.fill(line, width=width) for line in text.split('\n'))
        fig.text(x, y, wrapped, va='top', fontsize=size, linespacing=1.55, color=color)

    def picture(fig, t, bottom, height):
        ax = fig.add_axes([.045, bottom, .91, height])
        ax.imshow(Image.open(root/f'step_{t:02d}.png'))
        ax.axis('off')

    def save(fig, number):
        pdf.savefig(fig)
        fig.savefig(root/f'explanation_page_{number:02d}.png', dpi=120)
        plt.close(fig)

    fig = page(1, 'How our Craftax map remembers the world',
               'The first 15 actions of the trained map + MLP agent, shown with real observations.')
    cards = [('1. Observe', 'Read the current 7 × 9\nsymbolic terrain patch.'),
             ('2. Remember', 'Write terrain and timestamps\nat the correct map locations.'),
             ('3. Act', 'Read current state plus a\n33 × 33 remembered crop.')]
    for i, (title, body) in enumerate(cards):
        x = .055+i*.305
        fig.add_artist(FancyBboxPatch((x, .735), .275, .115, boxstyle='round,pad=.012', transform=fig.transFigure, facecolor='#eef5fa', edgecolor='none'))
        fig.text(x+.012, .817, title, fontweight='bold', fontsize=13, color=BLUE)
        fig.text(x+.012, .782, body, fontsize=10.5, va='top', linespacing=1.4)
    picture(fig, 0, .295, .405)
    paragraph(fig, .06, .265,
              'At reset, the agent is at world tile (x, y) = (32, 32). Its first view writes 63 terrain cells. '
              'All other cells remain unknown; the gray area is not copied from the simulator.', width=118)
    paragraph(fig, .06, .155,
              'After 15 actions, the agent has moved five tiles left and remembers 98 cells. '
              'The pictures render symbolic labels as pixels for us. The agent receives numbers, not these pictures.', width=118)
    save(fig, 1)

    fig = page(2, 'Displacement means movement that actually happened',
               'Positions are measured in tiles. They are not pixel positions or intended actions.')
    paragraph(fig, .06, .845,
              'We write coordinates as (x, y): x is the map column, increasing to the right; y is the row, '
              'increasing downward. The simulator stores the same coordinates in (row, column) order.', width=118)
    fig.text(.06, .725, r'$\mathrm{displacement} = \mathrm{position}_{t+1} - \mathrm{position}_t$', fontsize=19, color=BLUE)
    picture(fig, 7, .285, .405)
    paragraph(fig, .06, .25,
              'Step 7: LEFT succeeds. Position changes from (32, 32) to (31, 32), so (dx, dy) = (-1, 0). '
              'The map shifts its readout accordingly and reveals a new column of seven cells.', width=118)
    paragraph(fig, .06, .14,
              'Step 5: DOWN was chosen, but position stayed (32, 32). Displacement was (0, 0). '
              'Using the chosen action alone would incorrectly move the map. We use actual simulator positions.', width=118)
    save(fig, 2)

    fig = page(3, 'How one observation updates the map',
               'A deterministic write operation aligns local observations with persistent world locations.')
    paragraph(fig, .06, .845,
              'The player is always at local column 4, row 3 of the 9-column × 7-row view. '
              'After step 7, its world position is (31, 32). Thus local (column, row) = (0, 0) maps to world (27, 29).', width=118)
    fig.text(.07, .72, r'$x_{\mathrm{world}} = x_{\mathrm{agent}} + c - 4 \qquad '
             r'y_{\mathrm{world}} = y_{\mathrm{agent}} + r - 3$', fontsize=18, color=BLUE)
    ax = fig.add_axes([.06, .315, .88, .345])
    ax.set_facecolor('#f1f5f8')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    code = '''displacement = current_simulator_position - previous_simulator_position
map_position = map_position + displacement

for each visible tile at local row r, column c:
    world_row = map_position.row + r - 3
    world_col = map_position.col + c - 4
    terrain[world_row, world_col] = observed_terrain_type
    last_seen[world_row, world_col] = current_step

# Cells outside the current view retain their stored terrain and timestamp.'''
    ax.text(.025, .94, code, va='top', fontsize=10.5, family='DejaVu Sans Mono', linespacing=1.45)
    paragraph(fig, .06, .265,
              'Before step 7: visible columns are x = 28…36. After moving left: visible columns are x = 27…35. '
              'Column 27 is new, column 36 leaves view, and overlapping cells are refreshed. Total memory: 63 → 70 cells.', width=118)
    paragraph(fig, .06, .145,
              'Terrain edits are also remembered: at step 14, tile (26, 32) changes from grass to a crafting table. '
              'The map overwrites that tile because the change is visible.', width=118)
    paragraph(fig, .06, .073,
              'Rendered view: each tile is 28 × 28 pixels. Local tile (c, r) has pixel center '
              '(28c + 14, 28r + 14); the player is centered at pixel (126, 98).', width=135, size=9.5, color=GRAY)
    save(fig, 3)

    fig = page(4, 'The complete first 15 actions',
               'Step 0 is the initial observation; each subsequent row is the state after the listed action.')
    headers = ['Step', 'Action', 'World (x, y)', '(dx, dy)', 'New cells', 'Remembered']
    cells = [[r['step'], r['action'].replace('_', ' ').lower(), f'({r["x"]}, {r["y"]})',
              f'({r["dx"]}, {r["dy"]})', r['new_cells'], r['remembered_cells']] for r in rows]
    ax = fig.add_axes([.055, .24, .89, .61])
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=headers, cellLoc='center',
                     colWidths=[.07, .28, .20, .17, .13, .15], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if r == 0:
            cell.set_facecolor('#173b58')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#e5f2fa' if int(rows[r-1]['new_cells']) > 0 else ('#f3f5f7' if r % 2 else 'white'))
    paragraph(fig, .06, .195,
              'Blue rows add previously unseen locations. Mining or building can change terrain without adding cells: '
              'at steps 6, 10 and 13, a visible tree becomes grass; at step 14, grass becomes a crafting table.', width=118)
    paragraph(fig, .06, .09,
              'The failed DOWN move at step 5 adds no cells. Repeated observation still refreshes timestamps.', width=118, size=10.5)
    save(fig, 4)

    fig = page(5, 'Terrain says what; age says when it was seen',
               'The two grids refer to the same locations. The policy learns how to use their combination.')
    picture(fig, 15, .425, .405)
    x, y = 36, 32
    code = int(data['terrain'][15, y, x])
    terrain_name = metadata['terrain_codes'][str(code)].lower()
    last = int(data['last_seen'][15, y, x])
    age = 15-last
    scaled = np.log1p(age)/np.log(10001)
    paragraph(fig, .06, .395,
              f'Example: tile ({x}, {y}) is a {terrain_name}. It was last visible at step {last}. '
              f'At step 15 the stored terrain is unchanged, but its age is {age} steps '
              f'(scaled network input ≈ {scaled:.3f}). The bright current-view region has age zero.', width=118)
    fig.text(.06, .265, r'$\mathrm{age}=t-\mathrm{last\_seen}, \qquad '
             r'\mathrm{scaled\ age}=\frac{\log(1+\min(\mathrm{age},10000))}{\log(10001)}$', fontsize=16, color=BLUE)
    paragraph(fig, .06, .205,
              'The network receives the original 1,345 numbers + 1,089 remembered terrain IDs + 1,089 ages = 3,523 numbers. '
              'Terrain IDs become learned four-dimensional embeddings; the MLP then reads 6,790 features.', width=118)
    paragraph(fig, .06, .095,
              'The backing map persists for the episode; the policy sees a centered 33 × 33 crop. These figures use fixed '
              'world coordinates to make updates visible. No enemy tracks or unseen terrain are added. Memory clears on reset.', width=118, size=10.5)
    save(fig, 5)
    pdf.close()
    (root/'README.md').write_text('''# Map-memory walkthrough

Open `map_memory_explained.pdf` for the five-page illustrated explanation.
`index.html` provides a slider over all 16 states; `map_updates.gif` animates them.
`map_updates_first_15_steps.pdf` contains one full figure per state.

This is a real frozen map + MLP rollout: checkpoint 9,999,941,632, seed 18151,
episode index 0, standard action sampling (temperature 1). It contains the initial
observation and 15 transitions. `trajectory.csv`, `map_snapshots.npz`, and
`metadata.json` record the underlying data and validation.

The figures show fixed world coordinates for explanation; the actual policy
receives agent-centered 33×33 terrain and age crops. Pixels are rendered for
human inspection; memory writes use symbolic terrain labels and explicit actual
displacement from simulator coordinates. No unseen terrain is copied.

Regenerate the trace with `concept_mapping/explain_map_updates.py`; rebuild this
PDF with `concept_mapping/build_map_walkthrough_pdf.py --trace <this-folder>`.
The implementation being explained is `symbolic_map_memory.py` and its learned
reader is `models/actor_critic_map_mlp.py`. The trace validates visible writes,
timestamps, position alignment and preservation outside the current view.
''')
    print(output)


if __name__ == '__main__':
    main()
