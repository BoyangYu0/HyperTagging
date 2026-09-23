import pytest
from scripts.recover_reconstruction_phase57_diagnostics import validate_diagnostic_evidence


def evidence():return {'status':'TRAINED_COHORT_DEVIATION_DIAGNOSTIC_ONLY','strict_selection_overlap':100,'independent_validation':False,'original_receipt_status':'failed','contract_sha256':'bound'}


def test_explicit_contaminated_evidence_is_accepted():
    validate_diagnostic_evidence(evidence(),{'contract_sha256':'bound'})


@pytest.mark.parametrize('key,value',[('status','training_completed'),('strict_selection_overlap',0),('independent_validation',True),('original_receipt_status','completed'),('contract_sha256','unbound')])
def test_recovery_rejects_clean_relabelling_or_unbound_results(key,value):
    r=evidence();r[key]=value
    with pytest.raises(RuntimeError):validate_diagnostic_evidence(r,{'contract_sha256':'bound'})
