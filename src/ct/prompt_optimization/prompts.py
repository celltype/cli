"""
LLM prompt templates for mutation and evaluation in the prompt optimizer.
"""

# ── Mutation ─────────────────────────────────────────────────

MUTATION_SYSTEM = (
    "You are an expert at optimizing prompts for an AI drug discovery research agent. "
    "The agent has 190+ tools covering target discovery, compound profiling, expression analysis, "
    "viability screening, safety assessment, clinical positioning, and more. "
    "Effective prompts for this agent should:\n"
    "- Name specific entities (genes, compounds, cell lines, diseases, pathways)\n"
    "- Request quantitative data (IC50, EC50, p-values, effect sizes, selectivity windows)\n"
    "- Ask for mechanistic explanations (pathway-level, causal chains, not just facts)\n"
    "- Request actionable next steps (named assays, concentrations, cell lines, timelines)\n"
    "- Decompose complex questions into explicit sub-questions\n"
    "- Define success criteria (what a good answer must contain)\n"
    "- Consider drug-discovery-relevant dimensions: target validation, translational evidence, "
    "resistance/failure modes, assay strategy, risk assessment\n\n"
    "Your job: improve prompts while preserving the user's core research intent."
)

SEED_USER_TEMPLATE = (
    "Generate {count} diverse improved variants of this drug discovery research prompt. "
    "Each variant should take a different optimization angle:\n"
    "1. More specific — add named entities, quantitative requests, concrete endpoints\n"
    "2. More structured — decompose into numbered sub-questions with logical flow\n"
    "3. More mechanistic — ask for biological WHY, pathway explanations, causal reasoning\n"
    "{extra_angle}"
    "\nORIGINAL PROMPT:\n{prompt}\n\n"
    "Return a JSON array of {count} strings. Each string is a complete, improved prompt. "
    "No markdown, no explanation — only the JSON array."
)

MUTATE_USER_TEMPLATE = (
    "Improve this drug discovery research prompt. Apply ONE of these changes:\n"
    "- Add specificity (named targets, cell lines, concentrations, endpoints)\n"
    "- Add structure (break into numbered sub-questions)\n"
    "- Add mechanistic depth (ask for pathway-level explanations, causal chains)\n"
    "- Add actionability (request specific experimental follow-ups with details)\n"
    "- Add success criteria (define what a good answer must contain)\n"
    "- Remove ambiguity (replace vague terms with precise scientific language)\n\n"
    "PROMPT:\n{prompt}\n\n"
    "Return ONLY the improved prompt text, nothing else."
)

CROSSOVER_USER_TEMPLATE = (
    "Combine the best qualities of these two drug discovery research prompts into a single, "
    "superior prompt. Preserve the core research intent from both.\n\n"
    "PROMPT A:\n{prompt_a}\n\n"
    "PROMPT B:\n{prompt_b}\n\n"
    "Return ONLY the combined prompt text, nothing else."
)

# ── Evaluation ───────────────────────────────────────────────

EVAL_SYSTEM = (
    "You are an expert evaluator of drug discovery research prompts. "
    "You assess how well a prompt will elicit high-quality, actionable responses from an AI "
    "drug discovery agent with 190+ specialized tools. "
    "Good prompts produce answers that are complete, mechanistic, data-rich, and actionable."
)

EVAL_USER_TEMPLATE = (
    "Score this drug discovery prompt on 5 dimensions (0-3 each).\n\n"
    "PROMPT TO EVALUATE:\n{prompt}\n\n"
    "Rubric:\n\n"
    "1. **specificity** (0-3): Does it name specific entities (genes, compounds, cell lines, "
    "diseases)? Does it request quantitative data (IC50s, p-values, effect sizes)?\n"
    "   0 = completely vague, 3 = highly specific with named entities and quantitative requests\n\n"
    "2. **structure** (0-3): Is it decomposed into clear sub-questions? Does it have logical flow?\n"
    "   0 = single run-on question, 3 = well-organized multi-part with clear hierarchy\n\n"
    "3. **mechanistic_depth** (0-3): Does it ask for biological mechanisms, pathway-level "
    "explanations, causal chains? Does it go beyond surface-level facts?\n"
    "   0 = asks only for lists/facts, 3 = requests deep mechanistic insight and causal reasoning\n\n"
    "4. **actionability** (0-3): Does it request experimental next steps, specific assays, "
    "or concrete recommendations? Does it frame for drug discovery decisions?\n"
    "   0 = purely informational, 3 = explicitly requests actionable experiments with specifics\n\n"
    "5. **completeness** (0-3): Does it cover all relevant aspects? Would answering it fully "
    "require multi-disciplinary evidence (genetics + chemistry + clinical)?\n"
    "   0 = narrow/incomplete, 3 = comprehensive, multi-faceted, cross-disciplinary\n\n"
    "Return JSON with this exact structure (no markdown, no explanation):\n"
    '{{"scores": ['
    '{{"dimension": "specificity", "score": <0-3>, "reasoning": "<one sentence>"}},'
    '{{"dimension": "structure", "score": <0-3>, "reasoning": "<one sentence>"}},'
    '{{"dimension": "mechanistic_depth", "score": <0-3>, "reasoning": "<one sentence>"}},'
    '{{"dimension": "actionability", "score": <0-3>, "reasoning": "<one sentence>"}},'
    '{{"dimension": "completeness", "score": <0-3>, "reasoning": "<one sentence>"}}'
    "]}}"
)
