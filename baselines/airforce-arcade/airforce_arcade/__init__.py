from gym.envs.registration import register

envs_list = ['TestingRoomICRA-v0', 'DroneDodgeBall-v0', 'Refueling-v0',
             'TimedWaypoints-v0', 'CanyonRun-v0', 'DroneTag-v0', 'FireandIce-v0', 'DroneDuel-v0']

register(
    id='TestingRoomICRA-v0',
    entry_point='airforce_arcade.envs:TestingRoomICRA',
)

register(
    id='DroneDodgeBall-v0',
    entry_point='airforce_arcade.envs:DroneDodgeBall'
)

register(
    id='Refueling-v0',
    entry_point='airforce_arcade.envs:Refueling'
)

register(
    id='TimedWaypoints-v0',
    entry_point='airforce_arcade.envs:TimedWaypoints'
)

register(
    id='CanyonRun-v0',
    entry_point='airforce_arcade.envs:CanyonRun'
)

register(
    id='DroneTag-v0',
    entry_point='airforce_arcade.envs:DroneTag'
)

register(
    id='FireandIce-v0',
    entry_point='airforce_arcade.envs:FireandIce'
)

register(
    id='DroneDuel-v0',
    entry_point='airforce_arcade.envs:DroneDuel'
)
