import json
from dataclasses import replace

import pandas as pd
import pytest

from oribi import Evidence, Warning, inspect


@pytest.mark.parametrize("export", ["to_dict", "to_json"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_opt_in_structured_export_rejects_colliding_sample_keys(export, reverse, nested):
    entries = [(917, "numeric sample"), ("917", "text sample")]
    sample = dict(reversed(entries) if reverse else entries)
    if nested:
        sample = {"private parent key": [sample]}
    context = inspect(pd.DataFrame({"metadata": [sample]}))

    default = getattr(context, export)()
    if export == "to_json":
        default = json.loads(default)
    assert default["columns"][0]["statistics"]["top_values"] == []

    with pytest.raises(ValueError, match="duplicate stringified mapping key") as error:
        getattr(context, export)(include_values=True)
    message = str(error.value)
    assert "columns[0].statistics.top_values[0][0]" in message
    for private_text in ("917", "numeric sample", "text sample", "private parent key"):
        assert private_text not in message


def test_default_structured_export_preserves_nested_evidence_key_normalization():
    evidence = Evidence("example", "Example", {"counts": {1: 2, "1": 3}})
    warning = Warning("example", "info", "Example", "Example", evidence=(evidence,))
    context = replace(inspect(pd.DataFrame({"metadata": [1]})), warnings=(warning,))

    for exported in (context.to_dict(), json.loads(context.to_json())):
        details = exported["warnings"][0]["evidence"][0]["details"]
        assert details == {"counts": {"1": 3}}
