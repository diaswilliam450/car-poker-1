"""Bittensor configuration helpers vendored for the Poker44 subnet."""

from __future__ import annotations

import argparse

import bittensor as bt
import os
import traceback

traceback.format_exc()


def add_args(cls, parser: argparse.ArgumentParser) -> None:
    if parser is None:
        parser = argparse.ArgumentParser()
    bt.logging.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.Wallet.add_args(parser)
    bt.Axon.add_args(parser)
    
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=126)
    
    parser.add_argument(
        "--neuron.device",
        type=str,
        default="cpu",
        help="Torch device to execute forwards on (cpu, cuda:0, ...).",
    )
    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        default=50,
        help="Blocks between mandatory syncs.",
    )
    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Skip setting weights on-chain.",
    )
    parser.add_argument(
        "--neuron.wait_for_inclusion",
        action="store_true",
        default=True,
        help="Wait for weight-setting extrinsics to be included before treating them as successful.",
    )
    parser.add_argument(
        "--no-neuron.wait_for_inclusion",
        action="store_false",
        dest="neuron.wait_for_inclusion",
        help="Do not wait for inclusion when submitting weights.",
    )
    parser.add_argument(
        "--neuron.wait_for_finalization",
        action="store_true",
        default=True,
        help="Wait for weight-setting extrinsics to be finalized before treating them as successful.",
    )
    parser.add_argument(
        "--no-neuron.wait_for_finalization",
        action="store_false",
        dest="neuron.wait_for_finalization",
        help="Do not wait for finalization when submitting weights.",
    )
    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        default=0.05,
        help="Exponential moving average smoothing factor for scores.",
    )
    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        default=1,
        help="Concurrent forward coroutines to execute per step.",
    )
    parser.add_argument(
        "--neuron.timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for each validator to miner query.",
    )
    parser.add_argument(
        "--poll_interval_seconds",
        type=int,
        default=5 * 60,
        help="Default delay between validator ingestion cycles.",
    )
    parser.add_argument(
        "--neuron.axon_off",
        action="store_true",
        help="Disable serving the axon endpoint.",
    )
    parser.add_argument(
    "--blacklist.force_validator_permit",
    action="store_true",
    default=True,
    help="Only allow requests from validators with permits.",
    )
    parser.add_argument(
        "--no-blacklist.force_validator_permit",
        action="store_false",
        dest="blacklist.force_validator_permit",
        help="Allow registered callers without validator permits.",
    )
    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        default=False,
        help="Allow requests from non-registered entities.",
    )
    parser.add_argument(
        "--blacklist.allowed_validator_hotkeys",
        nargs="*",
        default=[],
        help="Optional allowlist of validator hotkeys permitted to query miners.",
    )
    parser.add_argument(
        "--wandb.off",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging for this neuron.",
    )
    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        default=False,
        help="Run Weights & Biases in offline mode.",
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="poker44-validators",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="",
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb.notes",
        type=str,
        default="",
        help="Optional notes to attach to the Weights & Biases run.",
    )

def add_validator_args(cls, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--neuron.name",
        type=str,
        default="validator",
        help="Trials go in neuron.root/(wallet_cold - wallet_hot)/neuron.name.",
    )
    parser.add_argument(
        "--validator.manual_players",
        nargs="*",
        default=[],
        help="Player descriptors to track manually (player_uid[:label]).",
    )


def add_miner_args(cls, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--neuron.name",
        type=str,
        default="miner",
        help="Trials go in neuron.root/(wallet_cold - wallet_hot)/neuron.name.",
    )
    parser.add_argument(
        "--miner.mock",
        action="store_true",
        help="Placeholder flag retained for compatibility.",
    )



def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    # if not config.neuron.dont_save_events:
    #     # Add custom event logger for the events.
    #     events_logger = setup_events_logger(
    #         config.neuron.full_path, config.neuron.events_retention_size
    #     )
    #     bt.logging.register_primary_logger(events_logger.name)


def _apply_cli_args(config: "bt.Config", parser: argparse.ArgumentParser) -> None:
    """Populate the Config from the parsed CLI args.

    bittensor >= 10 builds only its own namespaces (wallet/subtensor/axon/logging)
    from a parser and fills them with DEFAULTS — it neither parses the process CLI
    args nor nests custom dotted args (--netuid, --neuron.*, --blacklist.*). We
    re-apply the actually-parsed values so the subnet config surface
    (config.neuron.*, config.blacklist.*, config.netuid, plus CLI wallet/axon/
    subtensor overrides) resolves on every bittensor version (8.x and 10.x)."""
    parsed, _ = parser.parse_known_args()
    for dest, value in vars(parsed).items():
        if "." in dest:
            namespace, _, sub = dest.partition(".")
            current = config.get(namespace)
            if current is None:
                current = bt.Config()
                config[namespace] = current
            current[sub] = value
        else:
            config[dest] = value


def config(cls) -> bt.Config:
    parser = argparse.ArgumentParser()
    cls.add_args(parser)
    config = bt.Config(parser=parser)
    _apply_cli_args(config, parser)
    return config
