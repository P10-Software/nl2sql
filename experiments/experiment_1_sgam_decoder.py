import json

BIRD_TEST_DATA = ''
EHRSQL_TEST_DATA = ''
EHRSQL_DB = ''
TRIAL_TEST_DATA = ''
TRAIL_DB = ''


def prep_ehrsql():
    new_data = []
    schema = get_build_instructions(EHRSQL_DB, True)
    with open(EHRSQL_TEST_DATA, 'r') as fp:
        data = json.load(fp)
    for example in data:
        if example['gual_query'] is None:
            new_data.append({
                "id": example['id'],
                "question": example['question'],
                "feasible": 0,
                "schema": schema
            })
        else:
            new_data.append({
                "id": example['id'],
                "question": example['question'],
                "feasible": 1,
                "schema": schema
            })

    return new_data


def prep_trial():
    new_data = []
    schema = get_build_instructions(TRAIL_DB, True)
    with open(TRIAL_TEST_DATA, 'r') as fp:
        data = json.load(fp)
    for example in data:
        if example['gual_query'] is None:
            new_data.append({
                "id": example['id'],
                "question": example['question'],
                "feasible": 0,
                "schema": schema
            })
        else:
            new_data.append({
                "id": example['id'],
                "question": example['question'],
                "feasible": 1,
                "schema": schema
            })

    return new_data


def get_build_instructions(path, m_schema=False):
    raise NotImplementedError()


