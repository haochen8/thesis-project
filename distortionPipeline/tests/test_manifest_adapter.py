from src.pipeline.manifest_adapter import (
    augmented_manifest_to_clean_subset_datasets,
    augmented_manifest_to_deepfakebench_datasets,
    deepfakebench_dataset_to_pipeline_records,
    pipeline_records_to_deepfakebench_dataset,
)


def test_deepfakebench_roundtrip_preserves_metadata_and_frame_order():
    dataset = {
        "ToySet": {
            "real": {
                "train": {
                    "sample_a": {
                        "label": "real",
                        "frames": ["/data/real_a.jpg"],
                        "camera": "front",
                        "split_note": "keep-me",
                    },
                    "sample_b": {
                        "label": "real",
                        "frames": ["/data/real_b_1.jpg", "/data/real_b_2.jpg"],
                        "camera": "rear",
                    },
                }
            },
            "fake": {
                "val": {
                    "sample_c": {
                        "label": "fake",
                        "frames": ["/data/fake_c.jpg"],
                        "generator": "roop",
                    }
                }
            },
        }
    }

    records = list(deepfakebench_dataset_to_pipeline_records(dataset, dataset_name="ToySet"))

    assert [record["image_id"] for record in records] == [
        "sample_c",
        "sample_a",
        "sample_b__f0000",
        "sample_b__f0001",
    ]
    assert records[0]["source_metadata"]["generator"] == "roop"
    assert records[1]["source_metadata"]["split_note"] == "keep-me"

    rebuilt = pipeline_records_to_deepfakebench_dataset(records, dataset_name="ToySet")

    assert rebuilt == {
        "ToySet": {
            "fake": {
                "val": {
                    "sample_c": {
                        "label": "fake",
                        "frames": ["/data/fake_c.jpg"],
                        "generator": "roop",
                    }
                }
            },
            "real": {
                "train": {
                    "sample_a": {
                        "label": "real",
                        "frames": ["/data/real_a.jpg"],
                        "camera": "front",
                        "split_note": "keep-me",
                    },
                    "sample_b": {
                        "label": "real",
                        "frames": ["/data/real_b_1.jpg", "/data/real_b_2.jpg"],
                        "camera": "rear",
                    },
                }
            },
        }
    }


def test_augmented_manifest_creates_one_dataset_per_recipe_variant(tmp_path):
    input_jsonl = tmp_path / "jobs_with_paths.jsonl"
    input_jsonl.write_text(
        "\n".join(
            [
                '{"image_id":"sample_a","label":"real","recipe_id":"blur","recipe_instance_id":"blur__aaa","variant":0,"source_dataset_name":"ToySet","source_label":"real","source_split":"test","source_sample_id":"sample_a","source_frame_index":0,"source_frame_count":1,"source_frame_path":"/clean/a.jpg","distorted_path":"/distorted/a.png"}',
                '{"image_id":"sample_b","label":"fake","recipe_id":"blur","recipe_instance_id":"blur__aaa","variant":0,"source_dataset_name":"ToySet","source_label":"fake","source_split":"test","source_sample_id":"sample_b","source_frame_index":0,"source_frame_count":1,"source_frame_path":"/clean/b.jpg","distorted_path":"/distorted/b.png"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = augmented_manifest_to_deepfakebench_datasets(input_jsonl, tmp_path / "out")

    assert len(rows) == 1
    output_path = tmp_path / "out" / f"{rows[0]['dataset_name']}.json"
    rebuilt = output_path.read_text(encoding="utf-8")
    assert rows[0]["source_dataset_name"] == "ToySet"
    assert "/distorted/a.png" in rebuilt
    assert "/distorted/b.png" in rebuilt


def test_augmented_manifest_can_export_matched_clean_subset(tmp_path):
    input_jsonl = tmp_path / "jobs_with_paths.jsonl"
    input_jsonl.write_text(
        "\n".join(
            [
                '{"image_id":"sample_a","label":"real","recipe_id":"blur","recipe_instance_id":"blur__aaa","variant":0,"source_dataset_name":"ToySet","source_label":"real","source_split":"test","source_sample_id":"sample_a","source_frame_index":0,"source_frame_count":1,"source_frame_path":"/clean/a.jpg","distorted_path":"/distorted/a.png"}',
                '{"image_id":"sample_b","label":"fake","recipe_id":"blur","recipe_instance_id":"blur__aaa","variant":0,"source_dataset_name":"ToySet","source_label":"fake","source_split":"test","source_sample_id":"sample_b","source_frame_index":0,"source_frame_count":1,"source_frame_path":"/clean/b.jpg","distorted_path":"/distorted/b.png"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = augmented_manifest_to_clean_subset_datasets(input_jsonl, tmp_path / "out")

    assert len(rows) == 1
    output_path = tmp_path / "out" / f"{rows[0]['dataset_name']}.json"
    rebuilt = output_path.read_text(encoding="utf-8")
    assert rows[0]["source_dataset_name"] == "ToySet"
    assert "__clean_subset__" in rows[0]["dataset_name"]
    assert "/clean/a.jpg" in rebuilt
    assert "/clean/b.jpg" in rebuilt
    assert "/distorted/a.png" not in rebuilt
