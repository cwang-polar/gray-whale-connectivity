"""study render adapters; no analysis or scientific imports at import time."""

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
from shutil import copy2

from .control import digest
from .figure_adapters import plotting_imports


MODULES = {
    "Fig1": "build_figure1_three_panel_candidate_v2",
    "Fig2": "build_phase8t_robustness_candidates_v1",
    "Fig3": "build_ncc_seasonality_and_footprint_candidates",
    "Fig4": "build_phase8t_robustness_candidates_v1",
    "S1": "build_si_c1_annual_benthic_prey_observed_support_v1",
    "S2": "build_si_c3_consecutive_year_stability_v1",
    "S3": "build_si_c4_cost_observation_effort_sensitivity_v1",
    "S4": "build_si_b1_annual_sea_ice_accessibility_v1",
    "S5": "build_si_d1_aug_oct_corridor_observed_use_v1",
    "S6": "build_si_b2_june_july_asamm_observations_v1",
    "S7": "build_figure5_metric_sensitivity_v2",
    "S9": "build_si_b3_aug_oct_asamm_observations_v1",
    "S10": "build_figure5_selection_threshold_sensitivity_v4",
    "S11": "build_si_c5_pelagic_support_evidence_v2",
    "S12": "build_si_b4_pelagic_support_response_v1",
    "S13": "build_si_c2_prey_currency_sensitivity_v1",
}


@contextmanager
def bound_attributes(module, attributes):
    """Temporarily bind legacy path constants, restoring even after failure."""
    before = {name: getattr(module, name) for name in attributes}
    try:
        for name, value in attributes.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in before.items():
            setattr(module, name, value)


@contextmanager
def figure_outputs(module, paths):
    paths.require_reserved()
    root = paths.output("figures/" + paths.task)
    for folder in ("candidates", "source_data", "qa"):
        (root / folder).mkdir(parents=True, exist_ok=False)
    groups = {
        "": ["ROOT", "OUT_ROOT"],
        "candidates": ["CANDIDATE", "CANDIDATES", "CANDIDATE_OUT", "CANDIDATE_DIR"],
        "source_data": ["SOURCE", "SOURCE_DATA", "SOURCE_OUT", "SOURCE_DIR"],
        "qa": ["QA", "QA_OUT", "QA_DIR"],
    }
    bindings = {name: root / folder for folder, names in groups.items()
                for name in names if hasattr(module, name)}
    for name in ("LAND", "LAND_PATH"):
        if hasattr(module, name) and "land" in paths.contract["aliases"]:
            bindings[name] = paths.input("land")
    with bound_attributes(module, bindings):
        yield root


def geographic_polygons(frame):
    import geopandas as gpd
    return gpd.GeoDataFrame(frame.drop(columns="geometry_wkt"),
                            geometry=gpd.GeoSeries.from_wkt(frame.geometry_wkt), crs="EPSG:4326")


def reference_boxes(frame, base):
    import geopandas as gpd
    from shapely.geometry import Polygon
    boxes = gpd.GeoDataFrame([
        {"name": name, "geometry": Polygon(group.sort_values("vertex")[["longitude", "latitude"]].to_numpy())}
        for name, group in frame.groupby("reference_area", sort=True)
    ], crs="EPSG:4326")
    expected = base.make_dbo_reference_polygons().sort_values("name")
    if boxes.name.tolist() != expected.name.tolist() or not all(
            a.equals_exact(b, 1e-10) for a, b in zip(boxes.geometry, expected.geometry)):
        raise ValueError("Archived DBO vertices differ from the frozen drawing definition")
    return boxes


def cleanup_fig1_panel_c(ax):
    """Apply the publication labels to the stored schematic."""
    explanation = [t for t in ax.texts if t.get_text() == "cell value = proportion\nof paths crossing cell"]
    label = [t for t in ax.texts if t.get_text() == "frequency map"]
    cells = [p for p in ax.patches if type(p).__name__ == "Rectangle"]
    if len(explanation) != 1 or len(label) != 1 or len(cells) != 49:
        raise ValueError("Fig1C no longer matches the schematic label reference")
    center = (min(p.get_x() for p in cells) + max(p.get_x() + p.get_width() for p in cells)) / 2
    explanation[0].remove()
    label[0].set_x(center)
    label[0].set_horizontalalignment("center")


def observation_land(paths, extent):
    import geopandas as gpd
    from shapely.geometry import box
    land = gpd.read_file(paths.input("land")).to_crs("EPSG:4326")
    xmin, xmax, ymin, ymax = extent
    clip = box(xmin, ymin, xmax, ymax)
    land = land.loc[land.geometry.intersects(clip)].copy()
    land["geometry"] = land.geometry.intersection(clip).simplify(0.005, preserve_topology=True)
    return land.loc[~land.geometry.is_empty].copy()


def render_seasonality(module, paths, out, read):
    """Join the frozen panel tables without refitting or recalculating metrics."""
    import pandas as pd
    from pandas.testing import assert_frame_equal

    a = read("panel_A_total_cost")
    metrics = ["mean_path_cost_b", "percent_cost_change_b_vs_a", "jaccard_freq_gt_0_05",
               "centroid_displacement_km", "centroid_north_south_displacement_km"]
    ranges = a.drop(columns="prey_mean_path_cost")
    primary = a[["target_year", "prey_mean_path_cost", "mean_path_cost_b_primary"]].rename(
        columns={"prey_mean_path_cost": "mean_path_cost_a", "mean_path_cost_b_primary": "mean_path_cost_b"})
    late = a[["target_year"]].copy()
    for panel, metric, name in [("B", metrics[1], "mean_cost_percent_change"),
                                ("C", metrics[2], "jaccard_freq_gt_0_05"),
                                ("D", metrics[3], "centroid_shift_freq_gt_0_05_km")]:
        frame = read(f"panel_{panel}_seasonal_comparison")
        assert frame.target_year.tolist() == module.YEARS
        assert frame.reference.eq("same-year prey; frequent cells F_s > 0.05").all()
        ranges = ranges.merge(frame.filter(regex=f"target_year|{metric}"), on="target_year", validate="one_to_one")
        late[name] = frame.aug_oct_primary.to_numpy()
    joint = read("panels_A_E_all_early_anchor_values")
    e = read("panel_E_north_south")
    assert_frame_equal(e, joint[["setting", "target_year", metrics[-1]]])
    direction = module.legacy.range_table(e, module.PRIMARY, [metrics[-1]])
    ranges = ranges.merge(direction, on="target_year", validate="one_to_one")
    expected = module.legacy.range_table(joint, module.PRIMARY, metrics)
    assert_frame_equal(ranges[expected.columns], expected)
    identity = read("seasonal_input_identity_checks")
    assert identity.year.tolist() == module.YEARS
    assert identity.drop(columns="year").eq(True).all().all()
    grid = read("grid")
    cells = read("panel_F_map_cells")
    # The compact map export omits the static ocean mask, retained in the shared grid.
    cells = cells.merge(grid[["grid_id", "ocean_flag"]], on="grid_id", how="left", validate="many_to_one")
    assert cells.ocean_flag.eq(1).all()
    maps = {(season, int(year)): frame for (season, year), frame in cells.groupby(["season", "year"])}
    assert set(maps) == {(s, y) for s in ("May-July", "August-October") for y in module.MAP_YEARS}
    assert all(len(frame) == len(grid) and not frame.grid_id.duplicated().any() for frame in maps.values())

    class ArchivedBundle(module.Bundle):
        def __init__(self):
            self.root = out
            self.stem = Path(paths.figure_reference("Fig3")["source_path"]).stem
            self.outputs = []
            self.checks = {"archived_panel_tables_verified": True,
                           "seasonal_identity": "Existing recorded checks, not re-executed model checks"}

        def source(self, path, role):
            return paths.input_path(Path(path).relative_to(paths.data_root).as_posix())

    bundle = ArchivedBundle()
    with bound_attributes(module.legacy.fig3.BASE, {"LAND": paths.input("land")}), \
            bound_attributes(module, {"rel": paths.label}):
        module.style()
        module.draw_figure3(bundle, primary, late, ranges, grid, maps)
    (out / "qa/drawing_checks.json").write_text(json.dumps(bundle.checks, indent=2))


def render_archived_figure(paths):
    paths.require_task(*MODULES)
    protected = paths.snapshot()
    paths.reserve_output()
    with plotting_imports(paths):
        import pandas as pd
        import geopandas as gpd
        module = importlib.import_module(MODULES[paths.task])

        def read(alias):
            return pd.read_csv(paths.input(alias), dtype={"grid_id": str})

        with figure_outputs(module, paths) as out:
            task = paths.task
            if task == "Fig1":
                stations, endpoints = read("stations"), read("endpoints")
                if len(stations) != 1016 or not stations.DataYear.between(2007, 2019).all():
                    raise ValueError("Expected the 1,016 archived station plotting records")
                if endpoints.endpoint_type.value_counts().to_dict() != {"source": 40, "destination": 40}:
                    raise ValueError("Expected 40 archived endpoints of each type")
                grid = read("grid")
                grid = grid.loc[grid.ocean_flag.astype(int).eq(1)].copy()
                if len(grid) != 2839:
                    raise ValueError("Expected the frozen 2,839-cell ocean grid")
                prey = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude), crs=4326)
                prepared = (grid, reference_boxes(read("dbo_vertices"), module.base),
                            gpd.read_file(paths.input("land")).to_crs(4326), endpoints, prey)
                reference = paths.figure_reference("Fig1")
                revision = reference.get("display_revision", "legacy")
                if revision not in ("legacy", "label_cleanup_v3"):
                    raise ValueError("Unknown Fig1 display revision")
                original_panel_c = module.panel_c

                def panel_c(ax):
                    original_panel_c(ax)
                    if revision == "label_cleanup_v3":
                        cleanup_fig1_panel_c(ax)

                with bound_attributes(module.v1, {"CANDIDATE_OUT": out / "candidates"}), \
                        bound_attributes(module, {"panel_c": panel_c}):
                    module.build_figure(prepared=prepared, stem_name=Path(reference["source_path"]).stem)
            elif task == "Fig2":
                original_colorbar = module.fig2.add_colorbar_for_axes

                def frozen_colorbar(*args, **kwargs):
                    colorbar = original_colorbar(*args, **kwargs)
                    colorbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                    return colorbar

                with bound_attributes(module.fig2, {"ROOT": paths.data_root, "GRID": paths.input("grid"),
                        "LAND": paths.input("land"), "CENTROID_SUMMARY": paths.input("centroids"),
                        "add_colorbar_for_axes": frozen_colorbar}):
                    module.build_figure2(prepared_ranges=read("ranges"), show_range_annotation=False,
                        stem_name=Path(paths.figure_reference("Fig2")["source_path"]).stem)
            elif task == "Fig3":
                render_seasonality(module, paths, out, read)
            elif task == "Fig4":
                bindings = {"ANNUAL_INPUT": paths.input("annual"), "SPATIAL_INPUT": paths.input("spatial")}
                with bound_attributes(module.fig4, bindings), bound_attributes(module, {
                    "P3_ROUTE": paths.input("routes"), "P3_COMPARE": paths.input("comparisons")
                }):
                    annotation = importlib.import_module("build_phase8t_annotation_free_figures3_4_candidate_v1")
                    module.build_figure4(stem_name=annotation.FIG4_STEM,
                                         show_range_annotation=False, write_source_data=False)
            elif task == "S1":
                frame = module.load_source(paths.input("surface"))
                dbo = reference_boxes(read("dbo_vertices"), module.figure1_base)
                with bound_attributes(module, {"LAND_SHP": paths.input("land")}):
                    module.build_figure(prepared=(frame, dbo))
            elif task == "S2":
                module.configure_style()
                module.save_figure(module.draw_figure(read("stability")))
            elif task == "S3":
                module.configure_style()
                fig = module.draw_figure(read("costs"), read("summary"), read("grid"),
                                         read("masks"), geographic_polygons(read("dbo_wkt")))
                module.save_figure(fig)
            elif task == "S4":
                cells = read("cells")
                annual = {year: cells.loc[cells.target_year.eq(year)].copy() for year in module.YEARS}
                module.build_figure(annual, read("summary"), read("grid"))
            elif task == "S5":
                module.configure_style()
                module.build_figure(read("contrasts"), read("observed"))
            elif task == "S6":
                dbo = gpd.read_file(paths.input("dbo")).to_crs("EPSG:4326")
                prepared = (read("sightings"), read("summary"), read("effort"),
                            observation_land(paths, (-173.2, -156.4, 66.25, 72.55)),
                            dbo.loc[dbo.name.eq("DBO3")].copy(), read("boundary").sort_values("vertex_order"))
                module.build_figure(prepared=prepared)
            elif task == "S7":
                fig, _ = module.make_figure(read("effort"), read("topk"), read("distance"))
                module.save_figure(fig)
            elif task == "S9":
                prepared = (read("sightings"), read("effort"), read("summary"),
                            observation_land(paths, module.MAP_EXTENT), read("boundary").sort_values("vertex_order"))
                module.build_figure(prepared=prepared)
            elif task == "S10":
                module.configure_style()
                grid = read("grid")
                grid = grid.loc[grid.ocean_flag.astype(int).eq(1)].copy()
                module.build_figure(grid, read("summary"), read("stability"), read("response"))
            elif task == "S11":
                module.configure_style()
                cells = read("surface")
                surfaces = {year: cells.loc[cells.target_year.eq(year)].copy() for year in (2017, 2018)}
                fig = module.draw_figure(surfaces, read("observations"), read("models"), read("coverage"),
                                         gpd.read_file(paths.input("land")).to_crs(4326),
                                         geographic_polygons(read("dbo_wkt")))
                module.save_figure(fig)
            elif task == "S12":
                module.build_figure(prepared=(read("routes"), read("intervals"), read("observed")))
            elif task == "S13":
                module.build_figure(prepared=read("currencies"))
            module.plt.close("all")
            for relative in paths.contract["inputs"]:
                source = paths.input_path(relative)
                if source.suffix == ".csv":
                    target = out / "source_data" / source.name
                    if target.exists():
                        raise FileExistsError(f"Ambiguous source filename: {source.name}")
                    copy2(source, target)
            if task == 'S11' and paths.prepared_files:
                source = paths.input('observations')
                target = out / 'source_data' / source.name
                if target.exists():
                    raise FileExistsError('Prepared observation export would overwrite an input')
                copy2(source, target)
            (out / "caption.md").write_text(paths.figure_reference(task)["caption"], encoding="utf-8")
            if paths.snapshot() != protected:
                raise RuntimeError("An archived input changed during rendering")
            manifest = {"task": task, "layer": "archived source-data rendering, not raw-data reproduction",
                        "prepared_ierp_run": paths.prepared_ierp_run,
                        "inputs": protected, "drawing_module": paths.label(Path(module.__file__)),
                        "drawing_module_sha256": digest(Path(module.__file__)),
                        "outputs": [paths.label(p) for p in sorted(out.rglob("*")) if p.is_file()],
                        "source_precision": "Stored CSV precision retained; some historical exports were rounded.",
                        "models_or_observation_metrics_recalculated": False}
            (out / "qa/release_render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
