from dataclasses import dataclass
from pathlib import Path

import inspect
from typing import Literal
import typing

import yaml


# ref: https://qiita.com/kzmssk/items/f080c1487826222623eb
# ref: https://qiita.com/kzmssk/items/483f25f47e0ed10aa948
@dataclass
class YamlDataClass:
    def __post_init__(self):
        for key, val in self.__dataclass_fields__.items():
            member = getattr(self, key)
            if typing.get_origin(val.type) == Literal:
                print(f"Warn: Checking literal types not supported yet ({key})")
                continue
            elif typing.get_origin(val.type) == tuple:
                print(f"Warn: Checking tuple types not supported yet ({key})")
                continue
            assert isinstance(
                member, val.type
            ), f"Invalid Type of member ({key}): {type(member)} != {val.type}"

    @classmethod
    def load(cls, config_path: Path):
        """Load config from YAML file"""
        assert config_path.exists(), f"YAML config {config_path} does not exist"

        def convert_from_dict(parent_cls, data):
            for key, val in data.items():
                child_class = parent_cls.__dataclass_fields__[key].type
                if child_class == Path:
                    data[key] = Path(val)
                if inspect.isclass(child_class) and issubclass(
                    child_class, YamlDataClass
                ):
                    data[key] = child_class(**convert_from_dict(child_class, val))
            return data

        with open(config_path) as f:
            config_data = yaml.full_load(f)
            # recursively convert config item to YamlDataClass
            config_data = convert_from_dict(cls, config_data)
            return cls(**config_data)
