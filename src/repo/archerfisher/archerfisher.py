import os
from libs.archerfisher.src.benchmark.driver import Driver
from json import load
from yaml import safe_dump, safe_load
from src.common.logger import get_logger
from sqlglot import parse_one

logger = get_logger(__name__)


class ArcherFisher:
    """
    Interface for the ArcherFisher repository.
    Each instance handles one dataset, provided as a JSON file containing:
    {
        "id": 1,
        "question": "what doth life?",
        "goal_query": "SELECT meaning FROM life;"
    }
    """

    def __init__(self, dataset: str, name: str):
        self._check_repo()
        self.name = name
        self.workload_loc = None
        self.config_loc = None

        self._transform_dataset(dataset)
        self._create_config_file()

        self.driver = Driver(
            benchmark_config=self.config_loc,
            benchmark_name=self.name,
            workload_config=self.workload_loc,
            queries='',
            exclude_queries='',
            run_id='',
            schema_filter='',
            halt_on_error=True
        )

    def _check_repo(self):
        if not os.path.isdir('libs/archerfisher'):
            raise FileExistsError("Archerfisher submodule not found.")

    def _transform_dataset(self, dataset: str):
        try:
            with open(dataset, 'r') as f:
                data = load(f)

            archerfisher_dataset = {}

            for pair in data:
                tables, columns = self._find_tables_columns(pair['goal_query'])

                archerfisher_dataset[f'{self.name}_{pair["id"]}'] = {
                    "query_name": f'{self.name}_{pair["id"]}',
                    "question": pair['question'],
                    "golden_query": pair['goal_query'],
                    "tables": tables,
                    "comparison_rules": {
                        "columns": columns,
                        "match": "exact"
                    },
                    "auto_select_schema": True
                }

            dataset_path = os.path.join(
                'libs', 'archerfisher', 'resources', 'workloads', 'dataset', 'dataset.yaml'
            )
            with open(dataset_path, 'w') as f:
                safe_dump(archerfisher_dataset, f)

            self.workload_loc = dataset_path

        except Exception as e:
            logger.exception(
                f"Exception caught while processing input dataset: {e}")
            raise

    def _find_tables_columns(self, query: str):
        tables = set()
        columns = set()

        def walk(node):
            if node.key == 'table' and node.args.get('this'):
                tables.add(node.args['this'].name)

            if node.key == 'column' and node.args.get('this'):
                columns.add(node.args['this'].name)

            for child in node.args.values():
                if isinstance(child, list):
                    for item in child:
                        if hasattr(item, 'args'):
                            walk(item)
                elif hasattr(child, 'args'):
                    walk(child)

        ast = parse_one(query)
        walk(ast)
        return list(tables), list(columns)

    def _create_config_file(self):
        try:
            config_path = os.path.join(
                'libs', 'archerfisher', 'resources', 'config', 'benchmark', 'benchmark_config.yaml'
            )
            with open(config_path) as f:
                data = safe_load(f)

            implementation = {
                "name": self.name,
                "class": "llama_70b_benchmark.Lama70BBenchmark",
                "description": "Blank",
                "databases": {
                    "source": {
                        "type": "postgresql",
                        "connection": None  # TODO: Use DB connection extractor
                    },
                    "target": {
                        "same_as_source": True
                    }
                }
            }

            if 'implementations' not in data:
                data['implementations'] = []

            data['implementations'].append(implementation)

            with open(config_path, 'w') as f:
                safe_dump(data, f)

            self.config_loc = config_path

        except Exception as e:
            logger.exception(
                f"Exception caught while creating ArcherFisher config file: {e}")
            raise
