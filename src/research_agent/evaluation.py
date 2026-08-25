"""Evaluation suite: check the path taken and the quality of the answer.

Two things are worth checking about a state machine like this one. First,
whether a question reaches the nodes it should - a correct answer produced by
the wrong path is luck, not design. Second, whether the answer is any good,
which is scored by the LLM judge in `lib.evaluation`.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

from .framework.evaluation import AgentEvaluator, EvaluationResult, TestCase

from .app import Application
from .workflow import ResearchAgent

# The questions the project asks for. The first three are covered by the
# collection, so their reference answers state what the collection can actually
# support - the Pokemon entry records the year, not the per-region dates.
HALO_QUESTION = "What is the genre of Halo Infinite?"
HALO_ANSWER = "First-person shooter (FPS)"

POKEMON_QUESTION = "When were Pokemon Gold and Silver released?"
POKEMON_ANSWER = "1999, on the Game Boy Color."

MARIO_QUESTION = "Which one was the first 3D platformer Mario game?"
MARIO_ANSWER = "Super Mario 64, released in 1996 for the Nintendo 64."

MORTAL_KOMBAT_QUESTION = "Was Mortal Kombat X released for Playstation 5?"
MORTAL_KOMBAT_ANSWER = (
    "No - Mortal Kombat X was released in 2015 for PlayStation 4, not as a "
    "native PlayStation 5 title, although it is playable on PS5 through "
    "backward compatibility."
)


@dataclass(frozen=True)
class WorkflowCase:
    """A test case plus the node path the state machine is expected to take."""

    case: TestCase
    expected_path: List[str]


def default_cases() -> List[WorkflowCase]:
    '''
    In plain English: the standard set of questions used to check the agent still
    works.

    Five questions, chosen so that between them they force the pipeline down every
    possible route: three that the local database can answer, one that it cannot,
    and then that same unanswerable question a second time - which should now come
    back from the cache without touching the internet. Asking it twice is the whole
    trick; it is what proves the cache is doing its job.

    Output: the list of cases, each carrying the question, a model answer to be
    judged against, and the exact route the question ought to take.
    '''
    """The suite covering every branch of the machine.

    Returns:
        Collection hit (x3), cache miss, and cache hit - the last two asking the
        same question twice, so the cache demonstrably spares the second search.
    """
    collection_path = ["retrieve", "evaluate", "answer"]

    return [
        WorkflowCase(
            case=TestCase(
                id="collection_answers",
                description="Genre of a game that is in the collection",
                user_query=HALO_QUESTION,
                expected_tools=["retrieve_game", "evaluate_retrieval"],
                reference_answer=HALO_ANSWER,
            ),
            # The collection answers it, so the cache is never consulted.
            expected_path=collection_path,
        ),
        WorkflowCase(
            case=TestCase(
                id="pokemon",
                description="Release year of a game that is in the collection",
                user_query=POKEMON_QUESTION,
                expected_tools=["retrieve_game", "evaluate_retrieval"],
                reference_answer=POKEMON_ANSWER,
            ),
            expected_path=collection_path,
        ),
        WorkflowCase(
            case=TestCase(
                id="mario",
                description="First 3D Mario platformer - the collection says so explicitly",
                user_query=MARIO_QUESTION,
                expected_tools=["retrieve_game", "evaluate_retrieval"],
                reference_answer=MARIO_ANSWER,
            ),
            expected_path=collection_path,
        ),
        WorkflowCase(
            case=TestCase(
                id="cache_miss",
                description="Question the collection cannot cover, asked the first time",
                user_query=MORTAL_KOMBAT_QUESTION,
                expected_tools=["game_web_search", "register_memory"],
                reference_answer=MORTAL_KOMBAT_ANSWER,
            ),
            expected_path=[
                "retrieve", "evaluate", "recall", "web_search", "answer", "remember"
            ],
        ),
        WorkflowCase(
            case=TestCase(
                id="cache_hit",
                description="The same question again - from the cache, no web search",
                user_query=MORTAL_KOMBAT_QUESTION,
                expected_tools=["search_memory"],
                reference_answer=MORTAL_KOMBAT_ANSWER,
            ),
            expected_path=["retrieve", "evaluate", "recall", "answer"],
        ),
    ]


def run_case(
    agent: ResearchAgent,
    evaluator: AgentEvaluator,
    item: WorkflowCase,
    verbose: bool = True,
) -> Dict:
    '''
    In plain English: runs one test question and scores it two different ways.

    The first score is the route: did the question visit exactly the steps it was
    supposed to? This is a strict comparison, because the route is meant to be
    predictable - if it changes, something is genuinely broken.

    The second score is the answer itself, graded by a separate model call against a
    reference answer. This one is a judgement call, not a fact.

    Output: a small summary - which case, whether the route matched, the answer
    score, and how long it took. Collected into the table printed at the end.
    '''
    """Run one case, check the path it took, and judge the final answer.

    Args:
        agent: The agent under test.
        evaluator: Judge from `lib.evaluation`.
        item: The case and the path it is expected to take.
        verbose: Print a per-case report.

    Returns:
        A row for the summary table.
    """
    started = time.perf_counter()
    run = agent.invoke(item.case.user_query)
    elapsed = time.perf_counter() - started

    state = run.get_final_state()
    path = ResearchAgent.path_of(run)
    path_ok = path == item.expected_path

    judged: EvaluationResult = evaluator.evaluate_final_response(
        item.case, state["answer"], elapsed, 0
    )

    if verbose:
        print(f"\n=== {item.case.id} ===")
        print(f"question: {item.case.user_query}")
        print(f"path    : {' -> '.join(path)}")
        if not path_ok:
            print(f"expected: {' -> '.join(item.expected_path)}")
        print(f"cached  : {state.get('cached')}")
        print(f"answer  : {state['answer']}")
        print(f"path ok : {path_ok}   answer score: {judged.overall_score:.2f}")

    return {
        "id": item.case.id,
        "path_ok": path_ok,
        "answer_score": judged.overall_score,
        "seconds": elapsed,
    }


def run_suite(
    app: Application,
    cases: List[WorkflowCase] = None,
    verbose: bool = True,
) -> List[Dict]:
    '''
    In plain English: runs the whole set of test questions and prints the results.

    It empties the cache first. That matters more than it sounds: without it the
    "never seen this before" case would find last run's answer sitting in the cache,
    take the wrong route, and quietly stop testing anything.

    Output: one row per case, and a printed table. The rows are what the command
    line uses to decide the exit code - and only the route decides it, never the
    answer score, because two of the cases depend on whatever the live web returns
    that minute and would otherwise fail at random.
    '''
    """Run every case from a cold cache and print a summary table.

    The cache is cleared first: otherwise `cache_miss` would hit whatever a
    previous run left behind and take the `cache_hit` path instead, and the test
    would stop testing anything.

    Args:
        app: The assembled application.
        cases: Cases to run; the default suite if omitted.
        verbose: Print per-case reports and the summary.

    Returns:
        One row per case.
    """
    cases = cases if cases is not None else default_cases()
    evaluator = AgentEvaluator(
        model=app.settings.model, api_key=app.settings.openai_api_key
    )

    app.memory.clear(app.settings.memory_owner, app.settings.memory_namespace)

    results = [run_case(app.agent, evaluator, item, verbose) for item in cases]

    if verbose:
        print(f"\n{'=' * 64}\nSUMMARY")
        print(f"{'case':<20}{'path ok':>10}{'answer score':>14}{'seconds':>10}")
        for row in results:
            print(
                f"{row['id']:<20}{str(row['path_ok']):>10}"
                f"{row['answer_score']:>14.2f}{row['seconds']:>10.1f}"
            )

        by_id = {row["id"]: row for row in results}
        if {"cache_hit", "cache_miss"} <= by_id.keys():
            print(
                f"\ncache_hit answered in {by_id['cache_hit']['seconds']:.1f}s"
                f" vs {by_id['cache_miss']['seconds']:.1f}s for cache_miss"
            )

    return results
