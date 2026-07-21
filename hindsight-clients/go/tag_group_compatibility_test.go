package hindsight

import "testing"

func TestLegacyTagGroupTypesRemainAvailable(t *testing.T) {
	var _ MentalModelTriggerInputTagGroupsInner
	var _ MentalModelTriggerOutputTagGroupsInner
	var _ Not
	var _ Not1
}
