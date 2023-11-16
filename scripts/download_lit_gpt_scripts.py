import fsspec
from pathlib import Path

if __name__ == "__main__":
    destination = (
        Path(__file__).parent.parent / "sheeprlhf" / "utils" / "lit_gpt_scripts"
    )
    if destination.exists():
        if (
            input(
                f"Files already exist in {destination}. Do you want to overwrite them? [y/N] "
            ).lower()
            != "y"
        ):
            exit(0)
    destination.mkdir(exist_ok=True, parents=True)
    (destination / "__init__.py").touch()
    fs = fsspec.filesystem("github", org="Lightning-AI", repo="lit-gpt")
    fs.get(fs.ls("scripts/"), destination.as_posix())
