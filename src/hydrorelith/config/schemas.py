from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class StructureSourceConfig:
    pristine_structure_path: Path | None = None
    mpid: str | None = None
    target_ion: str = "Li"
    supercell: tuple[int, int, int] = (3, 3, 3)
    conventional_unit_cell: bool = True


@dataclass(slots=True)
class OutputConfig:
    output_dir: Path = Path("electrode_structures")
    output_format: str = "poscar"


@dataclass(slots=True)
class SamplingConfig:
    max_structures: int = 600
    oversampling_factor: int = 10
    min_lithiation_fraction: float = 0.75
    lithiation_step: float = 0.05
    max_removal_combinations_per_fraction: int = 200
    rattle_method: str = "mc"
    rattles_per_structure: int = 1
    rattle_std_300k: float = 0.01
    rattle_d_min: float = 1.5
    rattle_n_iter: int = 10
    max_base_structures_per_bin: int = 25
    phonon_fc2_path: Path | None = None
    phonon_qm_statistics: bool = False
    phonon_imag_freq_factor: float = 1.0
    rattle_engine: str = "hiphive"
    md_execution: str = "run"
    md_ensemble: str = "nvt"
    md_timestep_fs: float = 1.0
    md_steps: int = 500
    md_total_steps_budget: int = 60000
    md_sample_interval: int = 10
    md_frame_select_fraction: float = 0.10
    md_min_step_multiplier: float = 4.0
    md_friction_per_fs: float = 0.001
    uma_model_name: str = "uma-s-1p1"
    uma_task_id: str = "omat"
    uma_device: str = "cuda"
    matgl_model_name: str = "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES"
    matgl_backend: str = "dgl"
    direct_threshold_init: float = 0.05
    direct_plot_metrics: bool = True


@dataclass(slots=True)
class TemperatureAutoConfig:
    n_points: int = 5
    include_300k: bool = True
    melting_temperature_margin: float = 1.10


@dataclass(slots=True)
class TemperatureConfig:
    strategy: str = "fixed"
    values: tuple[int, ...] = (393, 433, 473, 493)
    pressures_mpa: dict[int, float] = field(default_factory=dict)
    auto: TemperatureAutoConfig = field(default_factory=TemperatureAutoConfig)


@dataclass(slots=True)
class RemovalVsRattleRatio:
    removal_combinations_fraction: float = 0.20
    rattled_variants_fraction: float = 0.80


@dataclass(slots=True)
class RetentionRatioConfig:
    temperature_weights: dict[int, float] = field(
        default_factory=lambda: {
            250: 0.15,
            300: 0.35,
            600: 0.25,
            900: 0.15,
            1200: 0.10,
        }
    )
    lithiation_bin_weights: dict[str, float] = field(
        default_factory=lambda: {
            "1.00-0.95": 0.30,
            "0.95-0.90": 0.25,
            "0.90-0.85": 0.20,
            "0.85-0.80": 0.15,
            "0.80-0.75": 0.10,
        }
    )


@dataclass(slots=True)
class RatioConfig:
    removal_vs_rattle: RemovalVsRattleRatio = field(default_factory=RemovalVsRattleRatio)
    removal_strategy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "symmetry_unique": 0.35,
            "random_uniform": 0.35,
            "dispersed_vacancies": 0.20,
            "clustered_vacancies": 0.10,
        }
    )
    retention: RetentionRatioConfig = field(default_factory=RetentionRatioConfig)


@dataclass(slots=True)
class ElectrodeGenerationConfig:
    source: StructureSourceConfig = field(default_factory=StructureSourceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    ratios: RatioConfig = field(default_factory=RatioConfig)
