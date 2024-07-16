import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='BertrandCompetitionDiscrete-v0',
    entry_point='gym_envs.envs.bertrand_competition_discrete:BertrandCompetitionDiscreteEnv',
    max_episode_steps=100,
)
