"""One factory for every stage of native SB3 Atari buyer training."""

from dataclasses import dataclass, replace

from stackelberg_pomdp.atari.core import SinglePlayerSpaceInvadersEnv
from stackelberg_pomdp.atari.wrappers import (
    AmmoLedger,
    AsymmetricBuyerObservationWrapper,
    AtariEpisodeMetricsWrapper,
    BulletMarketWrapper,
    ClipGameRewardWrapper,
    EpisodicLifeWrapper,
    FixedPriceProcess,
    FrameStackWrapper,
    MaxAndSkipWrapper,
    NoopResetWrapper,
    NoPriceProcess,
    ScarceAmmoWrapper,
    UniformPriceProcess,
    WarpFrameWrapper,
)


STAGES = {"gameplay", "free_trade", "priced", "fixed_price"}


@dataclass(frozen=True)
class AtariBuyerEnvConfig:
    stage: str = "gameplay"
    seed: int = 1
    initial_bullets: int = None
    bullet_capacity: int = 5
    offer_chances: int = 5
    max_purchases: int = 5
    price_min: float = 0.0
    price_max: float = 1.0
    fixed_price: float = 0.0
    noop_max: int = 30
    frame_skip: int = 4
    frame_stack: int = 4
    episodic_life: bool = True
    clip_game_rewards: bool = True
    max_steps: int = None
    max_frames: int = 100_000
    rom_path: str = None

    def resolved(self):
        if self.stage not in STAGES:
            raise ValueError(f"unknown Atari training stage: {self.stage}")
        initial = self.initial_bullets
        if initial is None:
            initial = 5 if self.stage == "gameplay" else 0
        return replace(self, initial_bullets=int(initial))


def _price_process(config):
    if config.stage == "gameplay":
        return NoPriceProcess()
    if config.stage == "free_trade":
        return FixedPriceProcess(0.0)
    if config.stage == "fixed_price":
        return FixedPriceProcess(config.fixed_price)
    return UniformPriceProcess(
        config.price_min,
        config.price_max,
        seed=config.seed + 811,
    )


def make_atari_buyer_env(config=None, **overrides):
    """Build the native Atari environment with identical spaces in all stages."""

    if config is None:
        config = AtariBuyerEnvConfig(**overrides)
    elif isinstance(config, dict):
        config = AtariBuyerEnvConfig(**{**config, **overrides})
    elif overrides:
        config = replace(config, **overrides)
    config = config.resolved()

    ledger = AmmoLedger(
        initial_ammo=config.initial_bullets,
        capacity=config.bullet_capacity,
    )
    env = SinglePlayerSpaceInvadersEnv(
        seed=config.seed,
        rom_path=config.rom_path,
        max_frames=config.max_frames,
    )
    env = NoopResetWrapper(env, noop_max=config.noop_max, seed=config.seed)
    if config.episodic_life:
        env = EpisodicLifeWrapper(env)
    if config.clip_game_rewards:
        env = ClipGameRewardWrapper(env)
    ammo_wrapper = ScarceAmmoWrapper(env, ledger=ledger)
    env = MaxAndSkipWrapper(ammo_wrapper, skip=config.frame_skip)
    env = WarpFrameWrapper(env)
    env = FrameStackWrapper(env, frames=config.frame_stack)
    market_wrapper = BulletMarketWrapper(
        env,
        ledger=ledger,
        price_process=_price_process(config),
        trade_enabled=config.stage != "gameplay",
        offer_chances=config.offer_chances,
        max_purchases=config.max_purchases,
        price_max=config.price_max,
    )
    env = AsymmetricBuyerObservationWrapper(
        market_wrapper,
        ledger=ledger,
        ammo_wrapper=ammo_wrapper,
        market_wrapper=market_wrapper,
    )
    env = AtariEpisodeMetricsWrapper(
        env,
        ledger=ledger,
        max_steps=config.max_steps,
    )
    env.atari_config = config
    env.ammo_ledger = ledger
    env.ammo_wrapper = ammo_wrapper
    env.market_wrapper = market_wrapper
    return env
