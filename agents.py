from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col


def smart_agent(obs_dict, config_dict):
    pass


def default_agent(obs_dict, config_dict=None):
    """This agent always moves toward observation.food[0] but does not take advantage of board wrapping"""
    observation = Observation(obs_dict)
    # configuration = Configuration(config_dict)
    configuration_columns = 11

    player_index = observation.index
    player_goose = observation.geese[player_index]
    player_head = player_goose[0]
    player_row, player_column = row_col(player_head, configuration_columns)

    food = observation.food[0]
    food_row, food_column = row_col(food, configuration_columns)

    if food_row > player_row:
        return Action.SOUTH.name
    if food_row < player_row:
        return Action.NORTH.name
    if food_column > player_column:
        return Action.EAST.name
    return Action.WEST.name
