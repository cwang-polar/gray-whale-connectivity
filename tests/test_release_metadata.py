"""Check the delivered metadata and entry contracts without scientific execution."""

import csv
import gzip
import json
from pathlib import Path
import re
import sys
import unittest

SOFTWARE = Path(__file__).resolve().parents[1]
DATA = SOFTWARE.parent / "dataset"
sys.path.insert(0, str(SOFTWARE / "src"))
from gray_whale_corridor import __version__
from gray_whale_corridor.release import control, ierp_inputs


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class ReleaseMetadataTests(unittest.TestCase):
    def test_version_matches_both_citations(self):
        for root in (SOFTWARE, DATA):
            value = re.search(r"^version:\s*['\"]?([^'\"\s]+)", (root / "CITATION.cff").read_text(), re.M)
            self.assertIsNotNone(value)
            self.assertEqual(value.group(1), __version__)

    def test_index_distinguishes_included_and_generated_sources(self):
        generated = []
        for row in read_rows(DATA / "source_data/v8_figure_table_index.csv"):
            if row["status"] == "generated_from_external_inputs":
                generated.append(row)
                self.assertEqual(row["v8_id"], "S11")
                self.assertEqual(row["staged_path"], "")
                self.assertIn("prepare-ierp", row["note"])
                self.assertIn("--prepared-ierp-run", row["note"])
            else:
                self.assertEqual(row["status"], "included")
                self.assertTrue((SOFTWARE.parent / row["staged_path"]).is_file(), row)
        self.assertEqual(len(generated), 1)

    def test_table_catalog_covers_the_data_archive(self):
        catalog = read_rows(DATA / "source_data/table_catalog_v1.csv")
        keys = [r["table"] for r in catalog]
        self.assertEqual(len(keys), len(set(keys)))
        actual = {"dataset/" + p.relative_to(DATA).as_posix() for p in DATA.rglob("*")
                  if p.is_file() and (p.suffix == ".csv" or p.name.endswith(".csv.gz"))}
        included = {r["table"] for r in catalog if r["delivery"].startswith("included")}
        self.assertEqual(included, actual)
        external = {"dataset/" + s["legacy_dataset_path"] for s in ierp_inputs.registry()["outputs"].values()}
        self.assertEqual({r["table"] for r in catalog if r["delivery"] == "generated_from_external_inputs"}, external)
        for row in catalog:
            if row["table"] in external:
                self.assertFalse((SOFTWARE.parent / row["table"]).exists())
                self.assertIn("prepare-ierp", row["preparation_entry"])
                continue
            path = SOFTWARE.parent / row["table"]
            with (gzip.open(path, "rt", newline="") if path.name.endswith(".gz") else path.open(newline="")) as stream:
                fields = next(csv.reader(stream))
            self.assertEqual(fields, json.loads(row["columns_json"]), row["table"])
            self.assertEqual(len(fields), int(row["field_count"]))
            if path in (DATA / "MANIFEST.csv", DATA / "source_data/table_catalog_v1.csv"):
                self.assertEqual(row["sha256"], "")
            else:
                self.assertEqual(control.digest(path), row["sha256"], row["table"])

    def test_portable_provenance_does_not_use_personal_roots(self):
        ice = list((DATA / "outputs/ice").glob("ice_accessibility_v1_*.csv"))
        self.assertEqual(len(ice), 36)
        for path in ice:
            for row in read_rows(path):
                value = row["source_sic_file"]
                self.assertTrue(value.startswith("data/raw/ice/"))
                self.assertFalse(Path(value).is_absolute())
        domain = json.loads((DATA / "data/processed/study_domain/study_domain_C3_recurring_support_v2.geojson").read_text())
        for feature in domain["features"]:
            self.assertTrue(feature["properties"]["source"].startswith("outputs/prey_v2/"))


if __name__ == "__main__":
    unittest.main()
