from .claims import Config as _Config


class Config(_Config):
    root_num_children: int = 1
    forest_name_suffix: str = _Config.forest_name_suffix + "-eval"

    llm_completion_kwargs = _Config.rollout_worker_config.llm_completion_kwargs.copy()
    rollout_worker_config = _Config.rollout_worker_config._replace(
        max_num_children=0, llm_completion_kwargs=llm_completion_kwargs
    )
