from scripts.generate_manifest import build_jobs


def test_build_jobs_preserves_source_metadata():
    dataset_records = [
        {
            "image_id": "sample_001",
            "label": "real",
            "path": "/data/sample_001.jpg",
            "source_dataset_name": "ToySet",
            "source_sample_id": "sample_001",
            "source_frame_index": 0,
            "camera": "front",
            "session": {"id": "sess-7"},
            "source_metadata": {
                "split_note": "original split metadata",
                "capture_device": "iphone",
            },
        }
    ]
    recipes = [
        {
            "recipe_id": "jpeg_compress_v1",
            "label": "jpeg",
            "steps": [{"name": "jpeg", "params": {"quality": 80}}],
        }
    ]

    jobs = build_jobs(dataset_records, recipes, global_seed=123, variants=1)

    assert len(jobs) == 1
    job = jobs[0]
    assert job["path"] == "/data/sample_001.jpg"
    assert job["src_path"] == "/data/sample_001.jpg"
    assert job["source_dataset_name"] == "ToySet"
    assert job["source_sample_id"] == "sample_001"
    assert job["source_frame_index"] == 0
    assert job["source_metadata"] == {
        "split_note": "original split metadata",
        "capture_device": "iphone",
        "camera": "front",
        "session": {"id": "sess-7"},
    }
