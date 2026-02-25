from __future__ import annotations

from hydrorelith.config.schemas import ElectrodeGenerationConfig


def default_electrode_generation_config() -> ElectrodeGenerationConfig:
    return ElectrodeGenerationConfig()
