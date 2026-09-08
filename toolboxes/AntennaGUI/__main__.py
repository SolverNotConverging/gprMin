"""Launch the optional desktop editor."""


def main():
    try:
        from .app import main as launch
    except ImportError as exc:
        raise SystemExit(f"Antenna GUI dependency missing: {exc}. See toolboxes/AntennaGUI/README.rst") from exc
    launch()


if __name__ == "__main__":
    main()
