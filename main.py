#!/usr/bin/env python3

import click
from tqdm import tqdm
import json
from pathlib import Path
import src.detect as detect
import src.colors as colors


@click.command()
@click.option(
    "-p",
    "--data_path",
    help="Path to data directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "-o",
    "--output_file_path",
    help="Path to output file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob("*.jpg")

    results = {}

    colors_to_count = colors.Color.Green | colors.Color.Red | colors.Color.Yellow | colors.Color.Purple

    for img_path in tqdm(sorted(img_list)):
        candy = detect.countCandy(str(img_path), colors_to_count)
        results[img_path.name] = candy

    with open(output_file_path, "w") as ofp:
        json.dump(results, ofp)


if __name__ == "__main__":
    main()
