def _parse_choices(response: str, n_choices: int):
    parsed = None
    try:
        parsed = response.split('\n- ')
        parsed[0] = '- '.join(parsed[0].split('- ')[1:])
        assert len(parsed) == n_choices, f'Response length: {len(parsed)}'
    except (IndexError, AttributeError, AssertionError) as e:
        print(f"Error parsing response: {e}\nResponse:\n{response}\nParsed choices: {parsed}")
        parsed = None
    
    # remove trailing spaces from responses
    if parsed is not None:
        parsed = [choice.rstrip() for choice in parsed]
    
    return parsed