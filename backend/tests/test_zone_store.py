import pytest

from app.schemas import ZoneCreate, ZonePoint, ZoneUpdate
from app.services.zone_store import ZoneStore


def new_zone(name="Entry"):
    return ZoneCreate(
        name=name,
        polygon=[
            ZonePoint(x=0.1, y=0.1),
            ZonePoint(x=0.9, y=0.1),
            ZonePoint(x=0.9, y=0.9),
        ],
    )


def test_zone_store_persists_multiple_camera_zones_and_updates(tmp_path):
    database = tmp_path / "zones.sqlite3"
    store = ZoneStore(database)
    store.initialize()
    first = store.create_zone("camera-a", new_zone("Front door"))
    second = store.create_zone("camera-a", new_zone("Hallway"))
    store.create_zone("camera-b", new_zone("Garage"))

    restarted_store = ZoneStore(database)
    restarted_store.initialize()
    assert [zone["name"] for zone in restarted_store.list_zones("camera-a")] == [
        "Front door",
        "Hallway",
    ]

    updated = restarted_store.update_zone(
        "camera-a",
        first["id"],
        ZoneUpdate(
            name="Entrance",
            polygon=[
                ZonePoint(x=0.2, y=0.2),
                ZonePoint(x=0.8, y=0.2),
                ZonePoint(x=0.5, y=0.8),
            ],
            enabled=False,
            loiter_seconds=45,
            event_cooldown_seconds=10,
        ),
    )
    assert updated is not None
    assert updated["enabled"] is False
    assert updated["loiter_seconds"] == 45
    assert restarted_store.delete_zone("camera-a", second["id"])
    assert len(restarted_store.list_zones("camera-a")) == 1
    assert len(restarted_store.list_zones("camera-b")) == 1


def test_zone_store_rejects_polygon_with_fewer_than_three_distinct_points(tmp_path):
    store = ZoneStore(tmp_path / "zones.sqlite3")
    store.initialize()
    repeated = ZoneCreate(
        name="Invalid",
        polygon=[
            ZonePoint(x=0.2, y=0.2),
            ZonePoint(x=0.2, y=0.2),
            ZonePoint(x=0.8, y=0.8),
        ],
    )
    with pytest.raises(ValueError, match="three distinct"):
        store.create_zone("camera-a", repeated)


def test_zone_store_removes_camera_zones(tmp_path):
    store = ZoneStore(tmp_path / "zones.sqlite3")
    store.initialize()
    store.create_zone("camera-a", new_zone())
    store.delete_camera_zones("camera-a")
    assert store.list_zones("camera-a") == []
