from dataclasses import dataclass
from typing import Self

from simple_parsing import (
    ArgumentGenerationMode,
    ArgumentParser,
    DashVariant,
    NestedMode,
    Serializable,
)
from simple_parsing.utils import Dataclass


def create_parser(configs: dict[str, type[Dataclass]]) -> ArgumentParser:
    """
    Create a parser with the given dataclasses added as arguments under the given names.

    Args:
        configs (Dict[str, Dataclass]): Dataclasses that define the arguments.

    Returns:
        parser (ArgumentParser): The parser with the defined arguments.
    """
    parser = ArgumentParser(
        add_option_string_dash_variants=DashVariant.DASH,
        argument_generation_mode=ArgumentGenerationMode.BOTH,
        add_config_path_arg=False,
        nested_mode=NestedMode.WITHOUT_ROOT,
    )
    for name, config in configs.items():
        parser.add_arguments(config, dest=name)
    return parser


@dataclass
class ConfigEntry(Serializable):
    """
    Convenience for the main config entry, which can be subclassed to get an easy way to
    create the parser from the config. (essentially a mixin).

    Also makes the serialisation easy, so it can be saved or loaded as/from a file.

    This assumes one main entry config and everything else is nested in this, rather
    than having them side by side.
    """

    @classmethod
    def create_parser(cls: type[Self], dest: str = "config") -> ArgumentParser:
        """
        Create the parser with this Dataclass as the main entry.

        Args:
            dest (str): Destination of the arguments in the parser
                [Default: "config"]

        Returns:
            parser (ArgumentParser): The parser with the defined arguments.
        """
        return create_parser({dest: cls})

    @classmethod
    def parse_config(cls: type[Self]) -> Self:
        """
        Parsers the options with this config and returns the instance of the config,
        i.e. the typed object with the arguments parsed into.

        Note: This is the type checked version of the argparse.Namespace, but only for
        the arguments defined in the dataclass.

        Returns:
            cfg (Self): The parsed config (instance of the dataclass config)
        """
        parser = cls.create_parser(dest="config")
        cfg: Self = parser.parse_args().config
        return cfg
