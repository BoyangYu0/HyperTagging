"""Bounded aggregate publication for the frozen Phase86 and cleanup review."""
import hashlib
import json

SOURCE = 'artifacts/codex/phase86_cleanup_review_20261010'
FILES = {'phase86-cleanup-summary.json'}
LIMIT = 32_768


def generate(root, output):
    """Authenticate the selected aggregate summary before publishing its bytes."""
    binding = json.loads((root / SOURCE / 'binding.json').read_text())
    path = root / SOURCE / 'summary.json'
    if binding['file'] != path.name or path.is_symlink() or path.stat().st_size > LIMIT:
        raise ValueError('Unsafe Phase86 summary')
    raw = path.read_bytes()
    if len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('Phase86 summary hash mismatch')
    value = json.loads(raw)
    phase = value['phase86']
    if (value['schema'] != 'phase86-cleanup-public-summary-v1'
            or phase['source_sha'] != binding['source_sha']
            or phase['source_set_gates_passed'] is not True
            or phase['status'] != 'COMPLETED_NO_CONTRAST_ADMITTED'
            or phase['events'] != 96 or phase['eligible_mothers'] != 455
            or sum(phase['first_failure_counts'].values()) != 455
            or phase['scientific_contrasts'] or phase['training_updates']
            or phase['fresh_development_events'] or phase['complete_evaluation']
            or phase['recursive_topology_pid_reachability_verified']):
        raise ValueError('Phase86 diagnostic scope changed')
    # Publish a named aggregate projection, not a byte-for-byte source mirror.
    # The source summary authenticates private report bindings; its digest is
    # sufficient public provenance without copying that transport registry.
    published = {
        'schema': 'phase86-cleanup-public-download-v1',
        'source_summary_sha256': binding['sha256'],
        'cutoff': value['cutoff'],
        'phase86': phase,
        'cleanup': value['cleanup'],
        'cr001': value['cr001'],
    }
    encoded = (json.dumps(published, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    (output / 'phase86-cleanup-summary.json').write_bytes(encoded)
    return phase | {'metric_download': {'filename': 'phase86-cleanup-summary.json',
                    'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}}


def render(value):
    """Explain source-set support without claiming recursive reachability."""
    return [
        'Phase86: structural support does not admit a contrast',
        '-----------------------------------------------------', '',
        'The 96-event TRAIN diagnostic retains all 455 eligible mothers. Frozen source-set gates pass,',
        'but both candidate branches have only four intrinsic-height-compatible deep fit targets.',
        'Height compatibility is necessary, not sufficient for exact recursive topology and PID.',
        'No structural or weighting contrast, training update or fresh development evaluation followed.', '',
        'Exact and inclusive B membership are zero of 64 B trials; continuum acceptance is zero of 64 events.',
        'Each category has only 16 events. This is not a complete evaluation or a performance improvement.', '',
        'See :doc:`../../phase86` for the full interpretation, cleanup accounting and frozen unfinished CR001 status.',
        ':download:`Selected aggregate evidence and source hashes <phase86-cleanup-summary.json>`.', '',
        'Cleanup reduced selected home logical bytes by 941,618,410 and allocated bytes by 254,145,536.',
        'Raw evidence is retained; no Git-history or measured quota reduction is claimed.', '',
    ]
