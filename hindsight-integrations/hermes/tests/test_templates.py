"""Starter templates applied by the setup wizard (#6471)."""

import re

from hindsight_hermes import templates


def _manifest(*triggers):
    return {"mental_models": [{"id": f"m{i}", "trigger": t} for i, t in enumerate(triggers)]}


def test_refresh_after_every_consolidation_becomes_an_hourly_schedule():
    manifest = templates.schedule_refreshes(
        _manifest({"refresh_after_consolidation": True, "mode": "full"}), "personal"
    )
    trigger = manifest["mental_models"][0]["trigger"]
    assert "refresh_after_consolidation" not in trigger
    assert re.fullmatch(r"([0-9]|[1-5][0-9]) \* \* \* \*", trigger["refresh_cron"])
    assert trigger["mode"] == "full"  # the rest of the template's trigger is kept


def test_each_model_gets_a_stable_minute_of_its_own():
    first = templates.schedule_refreshes(_manifest(*[{"refresh_after_consolidation": True}] * 3), "personal")
    again = templates.schedule_refreshes(_manifest(*[{"refresh_after_consolidation": True}] * 3), "personal")
    crons = [m["trigger"]["refresh_cron"] for m in first["mental_models"]]
    assert crons == [m["trigger"]["refresh_cron"] for m in again["mental_models"]]
    assert len(set(crons)) > 1


def test_a_template_that_already_schedules_or_stays_manual_is_left_alone():
    triggers = [{"refresh_cron": "0 3 * * *"}, {"refresh_after_consolidation": False}, {}]
    manifest = templates.schedule_refreshes(_manifest(*[dict(t) for t in triggers]), "personal")
    assert [m["trigger"] for m in manifest["mental_models"]] == triggers


def test_the_wizard_applies_the_scheduled_manifest(monkeypatch):
    sent = {}
    monkeypatch.setattr(templates, "fetch_hermes_templates", lambda: [{"id": "pa", "manifest_file": "x"}])
    monkeypatch.setattr(templates, "probe_existing_customization", lambda *a: False)
    monkeypatch.setattr(templates, "fetch_manifest", lambda e: _manifest({"refresh_after_consolidation": True}))
    monkeypatch.setattr(templates, "apply_template", lambda url, bank, key, manifest: sent.update(manifest))

    assert (
        templates.run_template_step(
            api_url="http://h", bank_id="b", api_key=None, select=lambda *a, **k: 0, cancelled=-1, log=lambda m: None
        )
        == "pa"
    )
    assert sent["mental_models"][0]["trigger"].keys() == {"refresh_cron"}
