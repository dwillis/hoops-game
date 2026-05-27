# PyInstaller spec for Hoops: WBB Edition
# Build: uv run pyinstaller hoops.spec --clean --noconfirm

from PyInstaller.utils.hooks import collect_all

block_cipher = None

polars_datas, polars_binaries, polars_hiddenimports = collect_all("polars")
pyarrow_datas, pyarrow_binaries, pyarrow_hiddenimports = collect_all("pyarrow")

SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]

a = Analysis(
    ["src/hoops/cli.py"],
    pathex=["src"],
    binaries=polars_binaries + pyarrow_binaries,
    datas=[
        ("data/rules", "data/rules"),
        ("data/brackets", "data/brackets"),
        ("data/conf_tournaments", "data/conf_tournaments"),
        ("data/teams", "data/teams"),
        ("data/players", "data/players"),
        ("data/pbp_distributions", "data/pbp_distributions"),
        ("data/games", "data/games"),
        *[
            (f"data/raw/wbb/{s}/player_box.parquet", f"data/raw/wbb/{s}")
            for s in SEASONS
        ],
    ]
    + polars_datas
    + pyarrow_datas,
    hiddenimports=["hoops", "hoops.cli", "hoops.ui.app"]
    + polars_hiddenimports
    + pyarrow_hiddenimports,
    excludes=["scipy", "matplotlib", "tkinter"],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="hoops",
    debug=False,
    strip=False,
    upx=True,
    console=True,
)
