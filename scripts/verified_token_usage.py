"""Verify cumulative token transitions while ignoring unchanged usage broadcasts."""


def turn_usage(previous, current, raw_usage, new_events):
    if any(current.get(key) != value for key, value in raw_usage.items()):
        raise ValueError('CLI usage differs from archived session totals')
    keys = set(current)
    cursor = {key: previous.get(key, 0) for key in keys}
    for event in new_events:
        total = event['total_token_usage']
        if set(total) != keys:
            raise ValueError('Token counter schema changed')
        delta = {key: total[key] - cursor[key] for key in keys}
        if any(value < 0 for value in delta.values()):
            raise ValueError('Cumulative usage decreased')
        if any(delta.values()):
            if delta != {key: event['last_token_usage'].get(key, 0) for key in keys}:
                raise ValueError('Counter transition differs from last token usage')
        # The service can broadcast an unchanged total with the preceding request's
        # last_token_usage. That is a status event, not another billed request.
        cursor = dict(total)
    if cursor != current:
        raise ValueError('Final token event differs from session totals')
    result = {key: current[key] - previous.get(key, 0) for key in current}
    if result['total_tokens'] != result['input_tokens'] + result['output_tokens']:
        raise ValueError('Total tokens differ from input plus output')
    return result
