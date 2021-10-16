"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Fractal Fft."""


if __name__ == "__main__":
    main(prog_name="fractal-fft")  # pragma: no cover
