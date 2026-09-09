
Tests marked `campaign_artifacts(...)` inspect immutable production evidence
that is not distributed with a Git checkout. Missing declared files or Git
refs are reported as skips in portable CI; they are never replaced by synthetic
science results. On a validation host with the evidence mounted, run
`python -m pytest --require-campaign-artifacts` to make missing prerequisites
fail. An available but corrupt artifact still fails its original assertions.
Ownership fixtures bind their expected UID to the test process; production
ownership constants remain unchanged.
