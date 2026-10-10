"""The compact API inventory retains full coverage and rejects unsafe content."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'docs/_ext'))
from wiki_inventory_delivery import write_inventory_delivery
from wiki_phase72_compact import canonical


def test_inventory_roundtrip_and_determinism(tmp_path):
    inventory = {'symbols': [{'name': 'example', 'line': 0, 'missing': None}], 'modules': [], 'count': 1}
    write_inventory_delivery(tmp_path, inventory)
    first = (tmp_path / 'inventory-download.json').read_bytes()
    write_inventory_delivery(tmp_path, inventory)
    assert (tmp_path / 'inventory-download.json').read_bytes() == first
    subprocess.run([sys.executable, 'inventory-decoder.txt'], cwd=tmp_path, check=True)
    assert (tmp_path / 'inventory-decoded.json').read_bytes() == canonical(inventory)
    envelope = json.loads(first)
    envelope['decoded_sha256'] = '0' * 64
    (tmp_path / 'inventory-download.json').write_text(json.dumps(envelope))
    assert subprocess.run([sys.executable, 'inventory-decoder.txt'], cwd=tmp_path, capture_output=True).returncode


def test_private_inventory_rejected(tmp_path):
    with pytest.raises(ValueError, match='Private'):
        write_inventory_delivery(tmp_path, {'hostname': 'internal.invalid'})
