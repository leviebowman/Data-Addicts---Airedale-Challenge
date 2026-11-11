from pathlib import Path

import pandas as pd


KEY_COLUMN = "Serial Number"


def _columns_are_equal(left: pd.Series, right: pd.Series) -> bool:
    equal_values = left.eq(right)
    both_missing = left.isna() & right.isna()
    return bool((equal_values | both_missing).fillna(False).all())


def join_datasets(data_directory: Path) -> tuple[pd.DataFrame, list[str]]:
    input_files = sorted(data_directory.glob("Competition_Dataset*.xlsx"))
    if not input_files:
        raise FileNotFoundError(
            f"No Competition_Dataset*.xlsx files found in {data_directory}"
        )

    frames: list[pd.DataFrame] = []
    column_groups: dict[str, list[str]] = {}
    seen_columns: set[str] = set()
    used_names = {KEY_COLUMN}

    for path in input_files:
        frame = pd.read_excel(path)
        if KEY_COLUMN not in frame.columns:
            raise KeyError(f"{path.name} is missing the join column {KEY_COLUMN!r}")
        if frame[KEY_COLUMN].isna().any():
            raise ValueError(f"{path.name} contains missing {KEY_COLUMN!r} values")
        if frame[KEY_COLUMN].duplicated().any():
            raise ValueError(f"{path.name} contains duplicate {KEY_COLUMN!r} values")

        renamed_columns: dict[str, str] = {}
        for column in frame.columns:
            original_name = str(column)
            output_name = original_name
            if original_name != KEY_COLUMN and original_name in seen_columns:
                output_name = f"{original_name} [{path.stem}]"
                suffix = 2
                while output_name in used_names:
                    output_name = f"{original_name} [{path.stem} {suffix}]"
                    suffix += 1

            renamed_columns[column] = output_name
            used_names.add(output_name)
            if original_name != KEY_COLUMN:
                column_groups.setdefault(original_name, []).append(output_name)
                seen_columns.add(original_name)

        frames.append(frame.rename(columns=renamed_columns))

    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.merge(
            frame,
            on=KEY_COLUMN,
            how="outer",
            validate="one_to_one",
            sort=False,
        )

    columns_to_drop: list[str] = []
    for columns in column_groups.values():
        retained: list[str] = []
        for column in columns:
            if any(_columns_are_equal(joined[column], joined[kept]) for kept in retained):
                columns_to_drop.append(column)
            else:
                retained.append(column)

    return joined.drop(columns=columns_to_drop), columns_to_drop


def main() -> None:
    data_directory = Path(__file__).resolve().parent
    output_path = data_directory / "TotalJoin.xlsx"
    joined, removed_columns = join_datasets(data_directory)
    joined.to_excel(output_path, index=False)
    print(
        f"Joined datasets into {output_path.name}: "
        f"{len(joined)} rows, {len(joined.columns)} columns."
    )
    if removed_columns:
        print(f"Removed identical repeated columns: {', '.join(removed_columns)}")


if __name__ == "__main__":
    main()