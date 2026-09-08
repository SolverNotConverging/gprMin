"""Reproducible GPU-synchronized cached-scene benchmark (not a GUI FPS promise)."""

import argparse
import json
from pathlib import Path
import time


def run(frames=45):
    import numpy as np
    import psutil
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    sphere = pv.Sphere(theta_resolution=128, phi_resolution=64)
    try:
        for x in range(8):
            for y in range(8):
                actor = plotter.add_mesh(sphere, color="#509e83", render=False)
                actor.position = (x * 1.2, y * 1.2, 0)
        plotter.view_isometric()
        plotter.show(auto_close=False)
        samples = []
        memory = []
        for index in range(frames):
            before = time.perf_counter()
            plotter.camera.Azimuth(2)
            plotter.render_window.Render()
            plotter.render_window.WaitForCompletion()
            if index >= 5:
                samples.append((time.perf_counter() - before) * 1000)
                memory.append(psutil.Process().memory_info().rss)
        return dict(
            triangles=sphere.n_cells * 64,
            actors=64,
            resolution=[1920, 1080],
            method="offscreen camera orbit with GPU completion; 5 warmup frames excluded",
            median_ms=float(np.median(samples)),
            p95_ms=float(np.percentile(samples, 95)),
            peak_sampled_rss_bytes=max(memory),
            gpu_memory_bytes=None,
            capabilities=plotter.render_window.ReportCapabilities(),
        )
    finally:
        plotter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "capabilities"}, indent=2))


if __name__ == "__main__":
    main()
