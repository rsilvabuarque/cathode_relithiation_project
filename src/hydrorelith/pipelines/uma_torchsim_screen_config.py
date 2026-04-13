from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ScreenConfig:
    electrode_manifest: Path | None
    electrolyte_manifest: Path | None
    electrode_root: Path | None
    electrolyte_root: Path | None
    output_dir: Path
    phase: str
    stages: tuple[str, ...]
    model_name: str
    device: str
    ensemble: str
    timestep_ps: float
    dump_every_steps: int
    minimize_steps: int
    heat_steps: int
    heat_start_temperature_k: float
    retherm_steps_electrode: int
    retherm_steps_electrolyte: int
    prod_steps: int
    production_stages: int
    replicas: int
    base_seed: int
    compute_stress: bool
    experimental_rates_csv: Path | None
    electrode_reference_pristine: Path | None
    use_default_tp_grid: bool
    tp_grid_csv: Path | None
    o_h_covalent_cutoff_A: float
    li_o_cutoff_A: float
    vacancy_site_match_cutoff_A: float
    analysis_frame_stride: int | None
    export_2pt: bool
    plots: bool
    max_memory_scaler: float | None
    precision: str
    debug: bool


def default_tp_grid() -> list[tuple[float, float]]:
    # (temperature_C, pressure_MPa)
    return [(220.0, 2.02), (200.0, 1.32), (160.0, 0.46), (120.0, 0.08)]


def default_config() -> ScreenConfig:
    return ScreenConfig(
        electrode_manifest=None,
        electrolyte_manifest=None,
        electrode_root=None,
        electrolyte_root=None,
        output_dir=Path("."),
        phase="both",
        stages=("all",),
        model_name="uma-s-1p1",
        device="auto",
        ensemble="nvt",
        timestep_ps=0.001,
        dump_every_steps=2,
        minimize_steps=2000,
        heat_steps=10000,
        heat_start_temperature_k=1.0,
        retherm_steps_electrode=2000,
        retherm_steps_electrolyte=8000,
        prod_steps=25000,
        production_stages=15,
        replicas=3,
        base_seed=0,
        compute_stress=False,
        experimental_rates_csv=None,
        electrode_reference_pristine=None,
        use_default_tp_grid=False,
        tp_grid_csv=None,
        o_h_covalent_cutoff_A=1.25,
        li_o_cutoff_A=2.6,
        vacancy_site_match_cutoff_A=1.0,
        analysis_frame_stride=None,
        export_2pt=False,
        plots=False,
        max_memory_scaler=None,
        precision="float32",
        debug=False,
    )


def _normalize_stages(raw_stages: str) -> tuple[str, ...]:
    tokens = [token.strip().lower() for token in raw_stages.split(",") if token.strip()]
    if not tokens:
        return ("all",)
    valid = {"md", "analyze", "export-2pt", "regress", "plots", "all"}
    unknown = [token for token in tokens if token not in valid]
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}")
    if "all" in tokens:
        return ("all",)
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return tuple(deduped)


def parse_args_to_config(args) -> ScreenConfig:
    cfg = default_config()
    cfg.electrode_manifest = args.electrode_manifest
    cfg.electrolyte_manifest = args.electrolyte_manifest
    cfg.electrode_root = args.electrode_root
    cfg.electrolyte_root = args.electrolyte_root
    cfg.output_dir = args.output_dir
    cfg.phase = args.phase
    cfg.model_name = args.model_name
    cfg.device = args.device
    cfg.ensemble = args.ensemble
    cfg.timestep_ps = args.timestep_ps
    cfg.dump_every_steps = args.dump_every_steps
    cfg.minimize_steps = args.minimize_steps
    cfg.heat_steps = args.heat_steps
    cfg.heat_start_temperature_k = args.heat_start_temperature_k
    cfg.retherm_steps_electrode = args.retherm_steps_electrode
    cfg.retherm_steps_electrolyte = args.retherm_steps_electrolyte
    cfg.prod_steps = args.prod_steps
    cfg.production_stages = args.production_stages
    cfg.replicas = args.replicas
    cfg.base_seed = args.base_seed
    cfg.compute_stress = args.compute_stress
    cfg.experimental_rates_csv = args.experimental_rates_csv
    cfg.electrode_reference_pristine = args.electrode_reference_pristine
    cfg.use_default_tp_grid = args.use_default_tp_grid
    cfg.tp_grid_csv = args.tp_grid_csv
    cfg.o_h_covalent_cutoff_A = args.o_h_covalent_cutoff_A
    cfg.li_o_cutoff_A = args.li_o_cutoff_A
    cfg.vacancy_site_match_cutoff_A = args.vacancy_site_match_cutoff_A
    cfg.analysis_frame_stride = args.analysis_frame_stride
    cfg.export_2pt = args.export_2pt
    cfg.plots = args.plots
    cfg.max_memory_scaler = args.max_memory_scaler
    cfg.precision = args.precision
    cfg.debug = args.debug

    stages = _normalize_stages(args.stage)
    if args.analysis_only:
        stages = ("analyze", "regress", "plots")
    if args.md_only:
        stages = ("md",)
    if cfg.export_2pt and "all" not in stages and "export-2pt" not in stages:
        stages = tuple(list(stages) + ["export-2pt"])
    if cfg.plots and "all" not in stages and "plots" not in stages:
        stages = tuple(list(stages) + ["plots"])
    cfg.stages = stages

    if cfg.replicas <= 0:
        raise ValueError("--replicas must be > 0")
    if cfg.dump_every_steps <= 0:
        raise ValueError("--dump-every-steps must be > 0")
    if cfg.timestep_ps <= 0:
        raise ValueError("--timestep-ps must be > 0")
    if cfg.minimize_steps < 0:
        raise ValueError("--minimize-steps must be >= 0")
    if cfg.heat_steps < 0:
        raise ValueError("--heat-steps must be >= 0")
    if cfg.heat_start_temperature_k <= 0:
        raise ValueError("--heat-start-temperature-k must be > 0")
    if cfg.production_stages <= 0:
        raise ValueError("--production-stages must be > 0")
    if cfg.precision not in {"float32", "float64"}:
        raise ValueError("--precision must be float32 or float64")
    return cfg
