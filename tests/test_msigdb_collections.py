from __future__ import annotations

from pathlib import Path

import pytest

from npathway.data import datasets as ds


@pytest.mark.parametrize(
    ("species", "collection", "expected_name"),
    [
        ("human", "go_cc", "c5.go.cc.v2024.1.Hs.symbols.gmt"),
        ("human", "go_mf", "c5.go.mf.v2024.1.Hs.symbols.gmt"),
        ("human", "c2_cp", "c2.cp.v2024.1.Hs.symbols.gmt"),
        ("human", "c7", "c7.all.v2024.1.Hs.symbols.gmt"),
        ("human", "msigdb_reactome", "c2.cp.reactome.v2024.1.Hs.symbols.gmt"),
        ("human", "msigdb_wikipathways", "c2.cp.wikipathways.v2024.1.Hs.symbols.gmt"),
        ("mouse", "go_cc", "m5.go.cc.v2024.1.Mm.symbols.gmt"),
        ("mouse", "go_mf", "m5.go.mf.v2024.1.Mm.symbols.gmt"),
        ("mouse", "msigdb_reactome", "m2.cp.reactome.v2024.1.Mm.symbols.gmt"),
    ],
)
def test_download_msigdb_gmt_supports_extended_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    species: str,
    collection: str,
    expected_name: str,
) -> None:
    calls: list[tuple[str, Path, str]] = []

    def fake_download(url: str, dest: Path, description: str = "") -> Path:
        calls.append((url, dest, description))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("SET_A\tdesc\tGENE1\tGENE2\tGENE3\n")
        return dest

    monkeypatch.setattr(ds, "_download_file", fake_download)

    path = ds.download_msigdb_gmt(
        collection=collection,
        species=species,
        cache_dir=tmp_path,
    )

    assert path.name == expected_name
    assert calls, "download should have been invoked"
    assert calls[0][1].name == expected_name
    assert expected_name in calls[0][0]


def test_list_supported_msigdb_collections_reports_new_aliases() -> None:
    human = set(ds.list_supported_msigdb_collections("human"))
    mouse = set(ds.list_supported_msigdb_collections("mouse"))

    assert {"go_cc", "go_mf", "c2_cp", "c7", "msigdb_reactome", "msigdb_wikipathways"} <= human
    assert {"go_cc", "go_mf", "c2_cp", "msigdb_reactome", "msigdb_wikipathways"} <= mouse

