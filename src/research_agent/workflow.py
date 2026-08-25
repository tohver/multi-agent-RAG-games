"""The agent itself: five tools wired as the nodes of one state machine."""

from typing import Dict, List, Tuple, TypedDict

from .framework.llm import LLM
from .framework.messages import SystemMessage, UserMessage
from .framework.state_machine import EntryPoint, Run, StateMachine, Step, Termination

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
    useful: bool
    evaluation: str
    cached: List[str]
    web_answer: str
    sources: List[Dict]
    answer: str


class ResearchAgent:
    """The whole pipeline as one state machine.

    Each of the five tools is a node, and three conditions decide which nodes
    actually run::

        retrieve -> evaluate -+-[useful]-----------------------> answer
                              |                                    ^
                              +-[not useful]-> recall -+-[hit]-----+
                                                       |
                                                       +-[miss]-> web_search -+
                                                                              |
                        termination <- [no web search] <- answer <------------+
                                            ^                 |
                                            +-- remember <----+ [after web search]

    Long-term memory is a cache of answers that previously cost a web search, so
    it sits on the path to the web: consulted only when the collection cannot
    answer, written only when the web actually had to be called.

    The LLM never picks the tools; `evaluate_retrieval` and the cache lookup do.
    The same question therefore takes the same path every time, and that path is
    recorded in `run.snapshots`.
    """

    def __init__(self, tools: ToolSet, settings: Settings):
        '''
        In plain English: sets the agent up and draws the map it will follow.

        It keeps hold of the five tools, creates the one model connection used for
        writing final answers, and then builds the route - which step leads to which,
        and on what condition. The route is built once here, not per question, so every
        question travels the same wiring.

        Output: nothing returned; the finished agent is the result. From this point
        `invoke` is all a caller needs.
        '''
        """Wire the state machine around one tool set.

        Args:
            tools: The five tools, already bound to their clients.
            settings: Runtime configuration; supplies the model and temperature.
        """
        self.tools = tools
        self.settings = settings
        self.llm = LLM(
            model=settings.model,
            temperature=settings.answer_temperature,
            api_key=settings.openai_api_key,
        )
        self.workflow = self._create_state_machine()

    # --- nodes -----------------------------------------------------------

    def _retrieve(self, state: ResearchState) -> ResearchState:
        '''
        In plain English: the first stop for every question - search the local database.

        It hands the question to the search tool and puts whatever comes back into the
        shared notes that travel through the pipeline.

        Output: the retrieved documents, stored under `documents`. The next step reads
        them to decide whether they are any good.
        '''
        """Node: search the internal vector database."""
        return {"documents": self.tools.retrieve_game(query=state["question"])}

    def _evaluate(self, state: ResearchState) -> ResearchState:
        '''
        In plain English: asks the judge whether the search found something usable.

        This is the step that stops the agent answering from irrelevant documents, which
        is the classic quiet failure of this kind of system.

        Output: `useful` (true/false) and the reasoning behind it, added to the shared
        notes. The true/false is read immediately afterwards to choose the next step.
        '''
        """Node: let the LLM judge decide whether those documents suffice."""
        report = self.tools.evaluate_retrieval(
            question=state["question"], retrieved_docs=state["documents"]
        )
        return {"useful": report.useful, "evaluation": report.description}

    def _recall(self, state: ResearchState) -> ResearchState:
        '''
        In plain English: checks the cache before spending money on a web search.

        Only reached when the local database fell short. If this same question was
        answered from the internet recently, the answer is still on disk and there is no
        reason to search again.

        Output: the cached entry, or an empty list. Empty means nothing usable was
        found, which sends the question to the web.
        '''
        """Node: before paying for a web search, check whether we already did."""
        return {"cached": self.tools.search_memory(query=state["question"])}

    def _web_search(self, state: ResearchState) -> ResearchState:
        '''
        In plain English: goes to the internet, having exhausted everything cheaper.

        Reached only when neither the local database nor the cache could help. This is
        the slowest and most expensive step in the whole pipeline, which is exactly why
        two cheaper checks come first.

        Output: a summary of what the web said, plus the sources behind it. Both go into
        the shared notes - the summary for writing the answer, the sources so they can
        be shown and stored.
        '''
        """Node: fall back to the web when neither collection nor cache helps."""
        result = self.tools.game_web_search(question=state["question"])
        return {
            "web_answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        }

    def _context(self, state: ResearchState) -> Tuple[str, str]:
        '''
        In plain English: gathers up whatever the run managed to find, ready to be put
        in front of the model.

        By this point the question may have been answered from the database, from the
        cache, or from the web - and the answer step should not have to care which. This
        looks at what is actually in the notes and picks the right material, in priority
        order: good documents first, then a cached answer, then web results.

        Output: two pieces of text - a name for the source ("the internal game
        database", "a web search"...) and the material itself. The name goes into the
        prompt too, so the model knows what it is quoting rather than treating everything
        as equally authoritative.
        '''
        """Assemble whatever grounding this particular run actually collected.

        Returns:
            A `(origin, context)` pair; `origin` names the source for the prompt.
        """
        if state.get("useful"):
            return (
                "the internal game database",
                "\n".join(f"- {doc}" for doc in state["documents"]),
            )

        cached = state.get("cached") or []
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
        '''
        In plain English: writes the final answer in ordinary language.

        This is the only step that composes prose. It is given the question and whatever
        material was gathered, and told firmly to use nothing else - if the material does
        not contain the answer, it must say so rather than fall back on what the model
        happens to remember. That restriction is what keeps answers traceable to a
        source.

        Output: the answer text, stored in the shared notes. That is what the user
        finally sees, and what gets saved to the cache if a web search paid for it.
        '''
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
        '''
        In plain English: saves the answer, but only when it cost a web search.

        Answers that came from the local database are not worth storing - the database
        is already there and already fast. Answers already served from the cache are not
        stored either, or the cache would fill with copies of itself.

        Output: nothing. The effect is a new cache entry, which is what makes the same
        question free the next time.
        '''
        """Node: cache the answer this web search just cost us."""
        self.tools.register_memory(
            question=state["question"],
            answer=state["answer"],
            sources=state.get("sources", []),
        )
        return {}

    # --- wiring ----------------------------------------------------------

    def _create_state_machine(self) -> StateMachine[ResearchState]:
        '''
        In plain English: draws the map - which step follows which, and when.

        Every tool becomes a numbered stop, and three junctions decide the route. This
        is where the pipeline's shape actually lives. Reading this one method tells you
        everything about how a question can travel; the individual steps above only know
        their own small job.

        Note that the model never gets a say in the route. That is the central design
        choice of this project: the same question always takes the same path, which
        makes runs reproducible and lets tests assert on the route itself.

        Output: the assembled state machine, stored on the agent and used by every call
        to `invoke`.
        '''
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
            '''
            In plain English: the first junction - were the local documents good enough?

            Yes, go straight to writing the answer. No, start looking elsewhere, beginning
            with the cache.

            Output: the next step to run. Called automatically by the state machine after
            the judge has spoken.
            '''
            """Good documents answer directly; otherwise try the cache."""
            return answer if state.get("useful") else recall

        def after_recall(state: ResearchState) -> Step[ResearchState]:
            '''
            In plain English: the second junction - did we already answer this before?

            A hit means the answer is on disk and the run can skip the web entirely, which is
            the whole point of keeping the cache.

            Output: the next step to run - the answer step on a hit, the web search on a
            miss.
            '''
            """A cache hit spares us the web search."""
            return answer if state.get("cached") else web_search

        def after_answer(state: ResearchState) -> Step[ResearchState]:
            '''
            In plain English: the third junction - is this answer worth saving?

            Only if it came from a fresh web search. That is checked by looking for web
            results in the notes, which only the web search step ever puts there - so an
            answer served from the cache cannot accidentally save itself back.

            Output: the next step - either the save step, or the end of the run.
            '''
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
        '''
        In plain English: ask the agent a question. This is the front door.

        Everything else in this file is machinery; this is the one method a caller
        needs. It drops the question into a fresh set of notes and lets the state machine
        run until it reaches the end.

        Output: a `Run` object. The answer is inside it, and so is the record of every
        step taken - which is how both the CLI and the tests can show or check the route
        afterwards.
        '''
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
        '''
        In plain English: tells you which route a question actually took.

        The state machine records each step as it goes. This reads that record back and
        strips the internal start marker, leaving a readable trail like
        `retrieve -> evaluate -> answer`.

        Output: the list of step names. The CLI prints it so you can see where an answer
        came from, and the tests compare it against the route the question was supposed
        to take - which is how a broken junction gets caught.
        '''
        """Return the node ids a run visited, without the entry marker."""
        return [s.step_id for s in run.snapshots if s.step_id != "__entry__"]
