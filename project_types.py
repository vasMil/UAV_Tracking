from typing import Literal, get_args

from constants import EGO_UAV_NAME, LEADING_UAV_NAME

# Define a Status_t (type) so you may use a str to define the status when the coSimulator exits
# but have the status be an int as it is expected
Status_t = Literal["Error", "Running", "Time's up", "LeadingUAV lost", "EgoUAV and LeadingUAV collision", "EgoUAV collision", "LeadingUAV collision"]

# Make sure that if I update the config file in the future, I will be reminded to also update the messages
# defined is Status_t
if EGO_UAV_NAME != "EgoUAV" or LEADING_UAV_NAME != "LeadingUAV":
    Warning("Noticed that you changed the names of the EgoUAV but forgot to update the Status_t type defined in project_types.py, or maybe just this if statement!")

def _map_to_status_code(status: Status_t) -> int:
    """
    Maps the status to the appropriate integer code.
    - status = Error => -1
    - status = Running -> 0
    - status = Time's up -> 1
    - status = EgoUAV and LeadingUAV collision -> 2
    - status = EgoUAV collision -> 3
    - status = LeadingUAV collision -> 4

    Returns:
    The code
    """
    return get_args(Status_t).index(status) - 1

Filter_t = Literal["KF", "None"]
Motion_model_t = Literal["CA", "CV"]
