from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_generate_figures_module():
    module_path = Path("reports/academic_report/generate_figures.py")
    spec = importlib.util.spec_from_file_location("report_generate_figures", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_all_figures_writes_expected_pngs(tmp_path):
    module = _load_generate_figures_module()

    created = module.generate_all_figures(tmp_path)

    expected_names = {
        "final_price_distribution.png",
        "acceptance_by_context.png",
        "delta_price_by_context.png",
        "acceptance_by_price_bin.png",
        "profit_heatmap.png",
        "trip_spatial_distribution.png",
        "distance_duration_relationship.png",
        "driver_model_comparison.png",
        "customer_demand_curves.png",
        "driver_supply_curves.png",
        "optimal_prices_by_context.png",
    }
    assert {path.name for path in created} == expected_names
    for path in created:
        assert path.exists()
        assert path.stat().st_size > 10_000
