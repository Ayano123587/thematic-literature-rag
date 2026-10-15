"""Fail closed when a pretrained QA checkpoint would initialize new parameters."""


def validate_loading_info(info):
    if any(info.get(key) for key in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Pretrained QA checkpoint has missing, mismatched or erroneous parameters')
    expected_unused = {'bert.pooler.dense.weight', 'bert.pooler.dense.bias'}
    if set(info.get('unexpected_keys', [])) - expected_unused:
        raise ValueError('Unexpected pretrained checkpoint parameters')
