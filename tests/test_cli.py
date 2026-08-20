"""Unit tests for the CLI entry point (no network).

Covers every command path in harness/cli.py:
  - retrieval-eval (all docs, single doc, no-docs error)
  - extract (all docs, single doc, no-docs error)
  - eval (mock-run, all-models, default, no-runs error, regression pass/fail)
  - build_parser + main (no command, with command)
"""

import argparse
from unittest.mock import patch, MagicMock

from harness import cli


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


class TestRetrievalEvalCommand:
    def test_all_docs(self):
        with patch.object(cli, "get_retrieval_doc_ids", return_value=["actt1", "onc1"]), \
             patch.object(cli, "run_retrieval_eval", return_value={"strategies": []}) as run, \
             patch.object(cli, "write_eval_output", return_value="evals/retrieval.json") as write, \
             patch.object(cli, "print_eval_table", return_value="table") as table:
            rc = cli.cmd_retrieval_eval(_ns(doc="all", k=None))
            assert rc == 0
            run.assert_called_once()
            write.assert_called_once()
            table.assert_called_once()

    def test_single_doc(self):
        with patch.object(cli, "run_retrieval_eval", return_value={"strategies": []}) as run, \
             patch.object(cli, "write_eval_output", return_value="x") as write, \
             patch.object(cli, "print_eval_table", return_value="t"):
            rc = cli.cmd_retrieval_eval(_ns(doc="actt1", k=3))
            assert rc == 0
            # single doc path bypasses get_retrieval_doc_ids
            run.assert_called_once_with(doc_ids=["actt1"], k=3)

    def test_no_docs_errors(self):
        with patch.object(cli, "get_retrieval_doc_ids", return_value=[]):
            rc = cli.cmd_retrieval_eval(_ns(doc="all", k=None))
            assert rc == 1


class TestExtractCommand:
    def test_all_docs(self):
        run = {"model": "glm-5.2", "extractions": {"actt1": {"a": 1}}}
        with patch.object(cli, "get_golden_doc_ids", return_value=["actt1"]), \
             patch.object(cli, "run_extraction", return_value=run) as run_fn, \
             patch.object(cli, "save_run", return_value="evals/runs/glm-5.2.json"):
            rc = cli.cmd_extract(_ns(model="glm-5.2", doc="all"))
            assert rc == 0
            run_fn.assert_called_once_with(model="glm-5.2", doc_ids=["actt1"])

    def test_single_doc(self):
        run = {"model": "glm-5.2", "extractions": {"actt1": {"a": 1}}}
        with patch.object(cli, "run_extraction", return_value=run) as run_fn, \
             patch.object(cli, "save_run", return_value="x"):
            rc = cli.cmd_extract(_ns(model="glm-5.2", doc="actt1"))
            assert rc == 0
            run_fn.assert_called_once_with(model="glm-5.2", doc_ids=["actt1"])

    def test_no_docs_errors(self):
        with patch.object(cli, "get_golden_doc_ids", return_value=[]):
            rc = cli.cmd_extract(_ns(model="glm-5.2", doc="all"))
            assert rc == 1


class TestEvalCommand:
    def _summary(self):
        return {"glm-5.2": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "n_documents": 3}}

    def test_mock_run(self):
        with patch.object(cli, "load_schema", return_value={"documents": []}), \
             patch.object(cli, "list_run_models", return_value=["glm-5.2"]), \
             patch.object(cli, "evaluate_all_models", return_value={"glm-5.2": {}}) as ev, \
             patch.object(cli, "build_summary", return_value=self._summary()), \
             patch.object(cli, "write_results"), \
             patch.object(cli, "write_summary"):
            rc = cli.cmd_eval(_ns(mock_run=True, all_models=False, regression=False))
            assert rc == 0
            ev.assert_called_once()

    def test_mock_run_no_runs_errors(self):
        with patch.object(cli, "load_schema", return_value={"documents": []}), \
             patch.object(cli, "list_run_models", return_value=[]):
            rc = cli.cmd_eval(_ns(mock_run=True, all_models=False, regression=False))
            assert rc == 1

    def test_all_models(self):
        with patch.object(cli, "load_schema", return_value={"documents": []}), \
             patch.object(cli, "evaluate_all_models", return_value={"glm-5.2": {}}), \
             patch.object(cli, "build_summary", return_value=self._summary()), \
             patch.object(cli, "write_results"), \
             patch.object(cli, "write_summary"):
            rc = cli.cmd_eval(_ns(mock_run=False, all_models=True, regression=False))
            assert rc == 0

    def test_default_scores_all(self):
        with patch.object(cli, "load_schema", return_value={"documents": []}), \
             patch.object(cli, "evaluate_all_models", return_value={"glm-5.2": {}}), \
             patch.object(cli, "build_summary", return_value=self._summary()), \
             patch.object(cli, "write_results"), \
             patch.object(cli, "write_summary"):
            rc = cli.cmd_eval(_ns(mock_run=False, all_models=False, regression=False))
            assert rc == 0

    def test_no_results_errors(self):
        with patch.object(cli, "load_schema", return_value={"documents": []}), \
             patch.object(cli, "evaluate_all_models", return_value={}):
            rc = cli.cmd_eval(_ns(mock_run=False, all_models=True, regression=False))
            assert rc == 1

    def test_regression_pass(self):
        report = {"glm-5.2": {"pass": True, "current_f1": 0.8, "baseline_f1": 0.79, "delta": 0.01}}
        with patch.object(cli, "load_schema", return_value={"documents": []}), \
             patch.object(cli, "evaluate_all_models", return_value={"glm-5.2": {}}), \
             patch.object(cli, "build_summary", return_value=self._summary()), \
             patch.object(cli, "write_results"), \
             patch.object(cli, "write_summary"), \
             patch.object(cli, "load_baseline", return_value={"glm-5.2": {"f1": 0.79}}), \
             patch.object(cli, "check_regression", return_value=report):
            rc = cli.cmd_eval(_ns(mock_run=False, all_models=True, regression=True))
            assert rc == 0

    def test_regression_fail(self):
        report = {"glm-5.2": {"pass": False, "current_f1": 0.7, "baseline_f1": 0.8, "delta": -0.1}}
        with patch.object(cli, "load_schema", return_value={"documents": []}), \
             patch.object(cli, "evaluate_all_models", return_value={"glm-5.2": {}}), \
             patch.object(cli, "build_summary", return_value=self._summary()), \
             patch.object(cli, "write_results"), \
             patch.object(cli, "write_summary"), \
             patch.object(cli, "load_baseline", return_value={"glm-5.2": {"f1": 0.8}}), \
             patch.object(cli, "check_regression", return_value=report):
            rc = cli.cmd_eval(_ns(mock_run=False, all_models=True, regression=True))
            assert rc == 1

    def test_regression_no_baseline_warns(self):
        with patch.object(cli, "load_schema", return_value={"documents": []}), \
             patch.object(cli, "evaluate_all_models", return_value={"glm-5.2": {}}), \
             patch.object(cli, "build_summary", return_value=self._summary()), \
             patch.object(cli, "write_results"), \
             patch.object(cli, "write_summary"), \
             patch.object(cli, "load_baseline", return_value={}):
            rc = cli.cmd_eval(_ns(mock_run=False, all_models=True, regression=True))
            assert rc == 0


class TestParserAndMain:
    def test_build_parser_has_commands(self):
        parser = cli.build_parser()
        # subcommands are registered; parse a known command
        args = parser.parse_args(["retrieval-eval", "--doc", "actt1"])
        assert args.command == "retrieval-eval"
        assert args.doc == "actt1"

    def test_main_no_command_prints_help(self):
        with patch.object(cli, "build_parser") as bp:
            parser = MagicMock()
            parser.parse_args.return_value = _ns(command=None)
            bp.return_value = parser
            rc = cli.main()
            assert rc == 1
            parser.print_help.assert_called_once()

    def test_main_with_command(self):
        with patch.object(cli, "build_parser") as bp:
            parser = MagicMock()
            args = _ns(command="eval")
            args.func = MagicMock(return_value=0)
            parser.parse_args.return_value = args
            bp.return_value = parser
            rc = cli.main()
            assert rc == 0
            args.func.assert_called_once_with(args)

    def test_main_command_without_func(self):
        with patch.object(cli, "build_parser") as bp:
            parser = MagicMock()
            parser.parse_args.return_value = _ns(command="eval")
            bp.return_value = parser
            rc = cli.main()
            assert rc == 1
