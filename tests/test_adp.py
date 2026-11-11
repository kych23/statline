import pytest

from statline.data.adp import MIN_PLAYERS, clean_adp, name_key


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("Chris Godwin Jr.", "chris godwin"),
        ("Kyle Pitts Sr.", "kyle pitts"),
        ("A.J. Brown", "aj brown"),
        ("Ja'Marr Chase", "jamarr chase"),
        ("Amon-Ra St. Brown", "amonra st brown"),
        ("Kenneth Walker III", "kenneth walker"),
    ],
)
def test_name_key_normalizes_punctuation_and_suffixes(raw: str, key: str) -> None:
    assert name_key(raw) == key


def player(name: str, position: str, adp: float) -> dict:
    return {"name": name, "position": position, "team": "DET", "adp": adp}


def test_clean_adp_maps_positions_and_rejects_thin_boards() -> None:
    filler = [player(f"P{i}", "WR", float(i)) for i in range(MIN_PLAYERS)]
    payload = {"players": [*filler, player("Jake Bates", "PK", 150.0), player("X", "OL", 1.0)]}
    out = clean_adp(payload, 2026)
    assert set(out["position"]) == {"WR", "K"}
    assert out.loc[out["name"] == "Jake Bates", "name_key"].item() == "jake bates"

    with pytest.raises(ValueError, match="no usable ADP board for 2026"):
        clean_adp({"players": filler[:4]}, 2026)
