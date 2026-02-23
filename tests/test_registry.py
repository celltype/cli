"""Tests for the tool registry."""

import inspect
import pytest
import ct.tools as tools_mod
from ct.tools import ToolRegistry, Tool, registry, ensure_loaded


class TestToolDataclass:
    def test_tool_creation(self):
        def dummy(**kwargs):
            return {"result": "ok"}

        t = Tool(
            name="test.dummy",
            description="A test tool",
            category="test",
            function=dummy,
            parameters={"x": "an arg"},
            requires_data=["crispr"],
        )
        assert t.name == "test.dummy"
        assert t.category == "test"
        assert t.requires_data == ["crispr"]

    def test_tool_run(self):
        def adder(a=0, b=0, **kwargs):
            return {"sum": a + b}

        t = Tool(name="math.add", description="add", category="math", function=adder)
        result = t.run(a=3, b=4)
        assert result == {"sum": 7}

    def test_tool_defaults(self):
        t = Tool(name="x", description="x", category="x", function=lambda **kw: {})
        assert t.parameters == {}
        assert t.requires_data == []
        assert t.usage_guide == ""

    def test_tool_usage_guide(self):
        t = Tool(
            name="test.guided",
            description="A guided tool",
            category="test",
            function=lambda **kw: {},
            usage_guide="Use when you need to test something.",
        )
        assert t.usage_guide == "Use when you need to test something."


class TestToolRegistry:
    def setup_method(self):
        self.reg = ToolRegistry()

    def test_register_decorator(self):
        @self.reg.register(
            name="test.func",
            description="A test function",
            category="test",
            parameters={"x": "value"},
        )
        def my_func(x=1, **kwargs):
            return {"x": x}

        tool = self.reg.get_tool("test.func")
        assert tool is not None
        assert tool.name == "test.func"
        assert tool.category == "test"
        assert tool.description == "A test function"
        # Function is still callable directly
        assert my_func(x=5) == {"x": 5}

    def test_register_with_usage_guide(self):
        @self.reg.register(
            name="test.guided",
            description="A guided tool",
            category="test",
            usage_guide="Use when testing.",
        )
        def guided(**kw): return {}

        tool = self.reg.get_tool("test.guided")
        assert tool.usage_guide == "Use when testing."

    def test_get_tool_missing(self):
        assert self.reg.get_tool("nonexistent") is None

    def test_list_tools_all(self):
        @self.reg.register(name="b.tool", description="B", category="b")
        def b(**kw): return {}

        @self.reg.register(name="a.tool", description="A", category="a")
        def a(**kw): return {}

        tools = self.reg.list_tools()
        assert len(tools) == 2
        # Sorted by name
        assert tools[0].name == "a.tool"
        assert tools[1].name == "b.tool"

    def test_list_tools_by_category(self):
        @self.reg.register(name="cat1.a", description="", category="cat1")
        def a(**kw): return {}

        @self.reg.register(name="cat2.b", description="", category="cat2")
        def b(**kw): return {}

        @self.reg.register(name="cat1.c", description="", category="cat1")
        def c(**kw): return {}

        cat1 = self.reg.list_tools(category="cat1")
        assert len(cat1) == 2
        assert all(t.category == "cat1" for t in cat1)

    def test_categories(self):
        @self.reg.register(name="z.1", description="", category="z")
        def z(**kw): return {}

        @self.reg.register(name="a.1", description="", category="a")
        def a(**kw): return {}

        cats = self.reg.categories()
        assert cats == ["a", "z"]

    def test_tool_descriptions_for_llm(self):
        @self.reg.register(
            name="target.score",
            description="Score a target",
            category="target",
            parameters={"gene": "gene name", "threshold": "score threshold"},
        )
        def score(**kw): return {}

        desc = self.reg.tool_descriptions_for_llm()
        assert "## target" in desc
        assert "**target.score**" in desc
        assert "gene: gene name" in desc

    def test_tool_descriptions_include_usage_guide(self):
        @self.reg.register(
            name="target.guided",
            description="A guided tool",
            category="target",
            parameters={"x": "value"},
            usage_guide="Use when you need to validate a target.",
        )
        def guided(**kw): return {}

        desc = self.reg.tool_descriptions_for_llm()
        assert "USE WHEN: Use when you need to validate a target." in desc

    def test_tool_descriptions_exclude_categories(self):
        @self.reg.register(name="a.t1", description="A tool", category="a")
        def a_tool(**kw): return {}

        @self.reg.register(name="b.t2", description="B tool", category="b")
        def b_tool(**kw): return {}

        desc = self.reg.tool_descriptions_for_llm(exclude_categories={"b"})
        assert "## a" in desc
        assert "**a.t1**" in desc
        assert "## b" not in desc
        assert "**b.t2**" not in desc

    def test_tool_descriptions_exclude_tools(self):
        @self.reg.register(name="a.t1", description="A tool", category="a")
        def a_tool(**kw): return {}

        @self.reg.register(name="a.t2", description="B tool", category="a")
        def b_tool(**kw): return {}

        desc = self.reg.tool_descriptions_for_llm(exclude_tools={"a.t2"})
        assert "**a.t1**" in desc
        assert "**a.t2**" not in desc

    def test_register_with_requires_data(self):
        @self.reg.register(
            name="data.tool",
            description="Needs data",
            category="data",
            requires_data=["proteomics", "prism"],
        )
        def data_tool(**kw): return {}

        tool = self.reg.get_tool("data.tool")
        assert tool.requires_data == ["proteomics", "prism"]


class TestGlobalRegistry:
    """Test that the global registry loads all tool modules correctly."""

    def test_ensure_loaded(self):
        ensure_loaded()
        tools = registry.list_tools()
        assert len(tools) >= 80  # 51 original + 7 ours + 28 David's (ops, dna, regulatory, pk, intel, etc.)

    def test_all_categories_present(self):
        ensure_loaded()
        cats = registry.categories()
        expected = {
            "target", "structure", "chemistry", "expression",
            "viability", "biomarker", "combination", "clinical",
            "literature", "safety",
            "cro", "compute", "experiment", "notification",
            "code", "files", "claude", "ops", "dna",
            "cellxgene", "clue", "remote_data",
        }
        assert expected.issubset(set(cats)), f"Missing categories: {expected - set(cats)}"

    def test_all_tools_have_required_fields(self):
        ensure_loaded()
        for tool in registry.list_tools():
            assert tool.name, "Tool missing name"
            assert tool.description, f"Tool {tool.name} missing description"
            assert tool.category, f"Tool {tool.name} missing category"
            assert callable(tool.function), f"Tool {tool.name} function not callable"

    def test_tool_names_match_category(self):
        ensure_loaded()
        for tool in registry.list_tools():
            prefix = tool.name.split(".")[0]
            assert prefix == tool.category, (
                f"Tool {tool.name} has category '{tool.category}' "
                f"but name prefix is '{prefix}'"
            )

    def test_specific_tools_exist(self):
        ensure_loaded()
        expected_tools = [
            # Original tools
            "safety.antitarget_profile",
            "safety.classify",
            "safety.sall4_risk",
            "clinical.indication_map",
            "clinical.population_size",
            "clinical.tcga_stratify",
            "literature.pubmed_search",
            "literature.chembl_query",
            "literature.openalex_search",
            "combination.synergy_predict",
            "combination.synthetic_lethality",
            "combination.metabolic_vulnerability",
            # New CRO tools
            "cro.search",
            "cro.match_experiment",
            "cro.compare",
            "cro.draft_inquiry",
            "cro.send_inquiry",
            # New compute tools
            "compute.list_providers",
            "compute.estimate_cost",
            "compute.submit_job",
            "compute.job_status",
            # New experiment tools
            "experiment.design_assay",
            "experiment.estimate_timeline",
            "experiment.list_assays",
            # New notification tool
            "notification.send_email",
            # Code execution tool
            "code.execute",
            # File I/O tools
            "files.read_file",
            "files.write_report",
            "files.write_csv",
            "files.list_outputs",
            # Claude reasoning tools
            "claude.reason",
            "claude.compare",
            "claude.summarize",
            # CELLxGENE tools
            "cellxgene.gene_expression",
            "cellxgene.cell_type_markers",
            "cellxgene.dataset_search",
            # CLUE API tools
            "clue.connectivity_query",
            "clue.compound_signature",
            # Remote data tools
            "remote_data.query",
            "remote_data.list_datasets",
            # Research ops tools
            "ops.notebook_add",
            "ops.todo_add",
            "ops.workflow_save",
            # Regulatory submission tools
            "regulatory.cdisc_lint",
            "regulatory.define_xml_lint",
            "regulatory.submission_package_check",
            # PK analysis tools
            "pk.nca_basic",
            # Pharma intelligence tools
            "intel.pipeline_watch",
            "intel.competitor_snapshot",
            # Translational tools
            "translational.biomarker_readiness",
            # Report tools
            "report.pharma_brief",
            # DNA tools
            "dna.reverse_complement",
            "dna.primer_design",
            # Parity API tools
            "data_api.mygene_lookup",
            "data_api.pdbe_search",
            "literature.preprint_search",
            # Chemistry parity utility
            "chemistry.sa_score",
        ]
        for name in expected_tools:
            assert registry.get_tool(name) is not None, f"Tool {name} not registered"

    def test_pubmed_search_registry_binding(self):
        """Regression test: literature.pubmed_search must point to pubmed_search()."""
        ensure_loaded()
        tool = registry.get_tool("literature.pubmed_search")
        assert tool is not None
        assert tool.function.__name__ == "pubmed_search"
        sig = inspect.signature(tool.function)
        assert any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ), "pubmed_search should accept **kwargs for executor-injected args"

    def test_all_existing_tools_have_usage_guide(self):
        """All tools should have usage_guide strings."""
        ensure_loaded()
        missing = []
        for tool in registry.list_tools():
            if not tool.usage_guide:
                missing.append(tool.name)
        assert not missing, f"Tools missing usage_guide: {missing}"


class TestToolModuleLoading:
    def test_loader_continues_after_import_error(self, monkeypatch):
        calls = []

        def fake_import_module(name):
            calls.append(name)
            if name == "ct.tools.b":
                raise ImportError("simulated missing optional dependency")
            return object()

        monkeypatch.setattr(tools_mod.importlib, "import_module", fake_import_module)
        monkeypatch.setattr(tools_mod, "_TOOL_MODULES", ("a", "b", "c"))
        monkeypatch.setattr(tools_mod, "_loaded", False)
        monkeypatch.setattr(tools_mod, "_load_errors", {})

        tools_mod.ensure_loaded()

        assert calls == ["ct.tools.a", "ct.tools.b", "ct.tools.c"]
        assert tools_mod.tool_load_errors().get("b", "").startswith("simulated missing optional dependency")
