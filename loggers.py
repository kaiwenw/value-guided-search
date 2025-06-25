import torch
import numpy as np


def process_metrics(metrics: dict[str, float | torch.Tensor | np.ndarray]) -> dict[str, float]:
    processed_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item()
        processed_metrics[key] = value
    return processed_metrics


class AbstractLogger:
    def __init__(self, **kwargs):
        pass

    def log_metrics(self, metrics: dict[str, float | torch.Tensor | np.ndarray], step: int):
        raise NotImplementedError()

    @property
    def id(self):
        raise NotImplementedError()

    def close(self):
        pass


class DummyLogger(AbstractLogger):
    def __init__(self, **kwargs):
        pass

    def log_metrics(self, metrics: dict[str, float | torch.Tensor | np.ndarray], step: int):
        pass

    @property
    def id(self):
        return "dummy"

    def close(self):
        pass


class NeptuneScaleLogger(AbstractLogger):
    def __init__(self, **kwargs):
        import neptune_scale
        # https://docs-beta.neptune.ai/run/
        config = kwargs.pop("config", None)
        tags = kwargs.pop("tags", None)
        print(kwargs)
        self.run = neptune_scale.Run(**kwargs)
        if config and isinstance(config, dict):
            self.run.log_configs({k: v for k, v in config.items() if v is not None})
        if tags and isinstance(tags, list):
            self.run.add_tags(tags)

    def log_metrics(self, metrics, step):
        metrics = process_metrics(metrics)
        self.run.log_metrics(metrics, step)

    @property
    def id(self):
        return self.run._run_id

    def close(self):
        self.run.close()
