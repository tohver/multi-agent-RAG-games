"""The agent itself: five tools wired as the nodes of one state machine."""

from typing import Dict, List, Tuple, TypedDict

from .lib.llm import LLM
from .lib.messages import SystemMessage, UserMessage
from .lib.state_machine import EntryPoint, Run, StateMachine, Step, Termination

from .config import Settings
from .tools import ToolSet

ANSWER_INSTRUCTIONS = (
    "You are a research assistant for the video game industry. "
    "Use only the context provided - never your own prior knowledge. "
    "If the context cannot answer the question, reply exactly: "
    "'Information not found'. Keep answers short and factual, and be precise "
    "about release dates and platforms."
)


class ResearchState(TypedDict):
    """Everything the nodes read and write, in one place."""

    question: str
    documents: List[str]
    documents_sufficient: bool
    evaluation_reason: str
    cached_answers: List[str]
    web_answer: str
    sources: List[Dict]
    answer: str


class ResearchAgent:
    """The whole pipeline as one state machine, one node per tool::

        retrieve -> evaluate -> [sufficient] answer
                             -> recall -> [hit]  answer
                                       -> [miss] web_search -> answer -> remember

    The cache is consulted only when the collection cannot answer, and written
    only when the web was actually called. Routing is decided by
    `evaluate_retrieval` and the cache lookup, never by the model, so a question
    always takes the same path; that path is recorded in `run.snapshots`.
    """

    def __init__(self, tools: ToolSet, settings: Settings):
        """Wire the state machine around one tool set.

        Args:
            tools: The five tools, already bound to their clients.
            settings: Runtime configuration; supplies the model and temperature.
        """
        self.tools = tools
        self.settings = settings
        self.llm = LLM(
            model=settings.chat_model,
            temperature=settings.answer_temperature,
            api_key=settings.openai_api_key,
        )
        self.workflow = self._create_state_machine()

    # --- nodes -----------------------------------------------------------

    def _retrieve(self, state: ResearchState) -> ResearchState:
        """Node: search the internal vector database."""
        return {"documents": self.tools.retrieve_game(query=state["question"])}

    def _evaluate(self, state: ResearchState) -> ResearchState:
        """Node: let the LLM judge decide whether those documents suffice."""
        report = self.tools.evaluate_retrieval(
            question=state["question"], retrieved_docs=state["documents"]
        )
        return {
            "documents_sufficient": report.useful,
            "evaluation_reason": report.reason,
        }

    def _recall(self, state: ResearchState) -> ResearchState:
        """Node: before paying for a web search, check whether we already did."""
        return {"cached_answers": self.tools.search_memory(query=state["question"])}

    def _web_search(self, state: ResearchState) -> ResearchState:
        """Node: fall back to the web when neither collection nor cache helps."""
        result = self.tools.game_web_search(question=state["question"])
        return {
            "web_answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        }

    def _context(self, state: ResearchState) -> Tuple[str, str]:
        """Assemble whatever grounding this particular run actually collected.

        Returns:
            A `(origin, context)` pair; `origin` names the source for the prompt.
        """
        if state.get("documents_sufficient"):
            return (
                "the internal game database",
                "\n".join(f"- {doc}" for doc in state["documents"]),
            )

        cached = state.get("cached_answers") or []
        if cached:
            return (
                "a previously cached web answer",
                "\n".join(f"- {entry}" for entry in cached),
            )

        return (
            "a web search",
            "\n".join(
                [state.get("web_answer", "")]
                + [f"- {s['title']}: {s['url']}" for s in state.get("sources", [])]
            ),
        )

    def _answer(self, state: ResearchState) -> ResearchState:
        """Node: compose the final answer from whatever grounding was gathered."""
        origin, context = self._context(state)

        ai_message = self.llm.invoke(
            [
                SystemMessage(content=ANSWER_INSTRUCTIONS),
                UserMessage(
                    content=(
                        f"# Question:\n{state['question']}\n\n"
                        f"# Context (from {origin}):\n"
                        f"{context.strip() or '(nothing)'}\n\n"
                        "# Answer:"
                    )
                ),
            ]
        )
        return {"answer": ai_message.content}

    def _remember(self, state: ResearchState) -> ResearchState:
        """Node: cache the answer this web search just cost us."""
        self.tools.register_memory(
            question=state["question"],
            answer=state["answer"],
            sources=state.get("sources", []),
        )
        return {}

    # --- wiring ----------------------------------------------------------

    def _create_state_machine(self) -> StateMachine[ResearchState]:
        """Build the state machine, its nodes and its three conditions."""
        machine = StateMachine[ResearchState](ResearchState)

        entry = EntryPoint[ResearchState]()
        retrieve = Step[ResearchState]("retrieve", self._retrieve)
        evaluate = Step[ResearchState]("evaluate", self._evaluate)
        recall = Step[ResearchState]("recall", self._recall)
        web_search = Step[ResearchState]("web_search", self._web_search)
        answer = Step[ResearchState]("answer", self._answer)
        remember = Step[ResearchState]("remember", self._remember)
        termination = Termination[ResearchState]()

        machine.add_steps(
            [entry, retrieve, evaluate, recall, web_search, answer, remember, termination]
        )

        def after_evaluation(state: ResearchState) -> Step[ResearchState]:
            """Good documents answer directly; otherwise try the cache."""
            return answer if state.get("documents_sufficient") else recall

        def after_recall(state: ResearchState) -> Step[ResearchState]:
            """A cache hit spares us the web search."""
            return answer if state.get("cached_answers") else web_search

        def after_answer(state: ResearchState) -> Step[ResearchState]:
            """Only a fresh web search is worth caching - a hit is already cached."""
            return remember if state.get("web_answer") else termination

        machine.connect(entry, retrieve)
        machine.connect(retrieve, evaluate)
        machine.connect(evaluate, [answer, recall], after_evaluation)
        machine.connect(recall, [answer, web_search], after_recall)
        machine.connect(web_search, answer)
        machine.connect(answer, [remember, termination], after_answer)
        machine.connect(remember, termination)

        return machine

    # --- public API ------------------------------------------------------

    def invoke(self, question: str) -> Run:
        """Run the pipeline for one question.

        Args:
            question: The user's question.

        Returns:
            The `Run`, whose `get_final_state()` holds the answer and whose
            `snapshots` record the path taken.
        """
        return self.workflow.run({"question": question})

    @staticmethod
    def path_of(run: Run) -> List[str]:
        """Return the node ids a run visited, without the entry marker."""
        return [s.step_id for s in run.snapshots if s.step_id != "__entry__"]
