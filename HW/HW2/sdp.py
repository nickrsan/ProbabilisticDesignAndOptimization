import constants
import support



def main(time_step_size=constants.TIME_STEP_SIZE, time_horizon=constants.TIME_HORIZON):

	scenarios = support.get_scenarios()  # gets the scenario objects

	# backward SDP with