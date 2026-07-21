"""Pin the public tag-group model names emitted by older SDK releases."""

from hindsight_client_api.models.mental_model_trigger_input_tag_groups_inner import (
    MentalModelTriggerInputTagGroupsInner,
)
from hindsight_client_api.models.mental_model_trigger_output_tag_groups_inner import (
    MentalModelTriggerOutputTagGroupsInner,
)
from hindsight_client_api.models.model_not import ModelNot
from hindsight_client_api.models.not1 import Not1


def test_legacy_tag_group_models_remain_importable():
    assert MentalModelTriggerInputTagGroupsInner.__name__ == "MentalModelTriggerInputTagGroupsInner"
    assert MentalModelTriggerOutputTagGroupsInner.__name__ == "MentalModelTriggerOutputTagGroupsInner"
    assert ModelNot.__name__ == "ModelNot"
    assert Not1.__name__ == "Not1"
