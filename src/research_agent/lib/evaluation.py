"""LLM-as-a-judge scoring for a finished answer."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .llm import LLM
from .parsers import PydanticOutputParser

# Rough blended rate for gpt-4o-mini: $0.15 per 1M input and $0.60 per 1M output
# tokens, assumed 50/50. Good enough to compare runs, not to bill anyone.
COST_PER_TOKEN = (0.15 + 0.60) / 2 / 1_000_000


class TaskCompletionMetrics(BaseModel):
    """Whether the task was finished, and in how many steps."""

    task_completed: bool = Field(description="Whether the task was completed successfully")
    steps_taken: int = Field(description="Number of steps taken to complete the task")
    expected_steps: Optional[int] = Field(description="Expected number of steps", default=None)


class QualityControlMetrics(BaseModel):
    """Whether the output was well formed and followed the instructions."""

    format_correct: bool = Field(description="Whether output format is correct")
    instructions_followed: bool = Field(description="Whether prompt instructions were followed")


class SystemMetrics(BaseModel):
    """What the run cost in tokens, time and money."""

    total_tokens: int = Field(description="Total tokens used")
    execution_time: float = Field(description="Total execution time in seconds")
    cost_estimate: Optional[float] = Field(description="Estimated cost in USD", default=None)


class EvaluationResult(BaseModel):
    """The complete verdict on one run."""

    task_completion: TaskCompletionMetrics
    quality_control: QualityControlMetrics
    system_metrics: SystemMetrics
    overall_score: float = Field(description="Overall evaluation score (0-1)", ge=0, le=1)
    feedback: str = Field(description="Detailed feedback and recommendations")


class TestCase(BaseModel):
    """One question, with what a good answer would look like.

    Attributes:
        id: Short identifier used in reports.
        description: What this case is meant to exercise.
        user_query: The question to ask.
        expected_tools: Tools the run is expected to use, for callers that
            check the route as well as the answer.
        reference_answer: What a correct answer contains. State what the data
            can actually support, not the whole truth, or the judge will
            penalise the agent for gaps in its sources.
        max_steps: Optional ceiling on the number of steps.
        context: Anything else a caller wants to carry along.
    """

    id: str
    description: str
    user_query: str
    expected_tools: List[str] = Field(default_factory=list)
    reference_answer: Optional[str] = None
    max_steps: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


class JudgeEvaluation(BaseModel):
    """The judge's structured reply."""

    task_completed: bool = Field(description="Whether the task was completed successfully")
    format_correct: bool = Field(description="Whether output format is correct")
    instructions_followed: bool = Field(description="Whether prompt instructions were followed")
    explanation: str = Field(description="Brief explanation of the evaluation")


class AgentEvaluator:
    """Scores a final answer with a second model acting as judge.

    Deliberately black box: it sees the question, the reference answer and what
    the agent replied, but nothing about how the agent got there. Checking the
    route is the caller's job, because what counts as the right route depends on
    the agent.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        """Create the judge.

        Args:
            model: Chat model to judge with. Temperature is fixed at 0.0, so
                the same answer scores the same twice.
            api_key: OpenAI key; falls back to the environment when omitted.
        """
        self.llm_judge = LLM(model=model, temperature=0.0, api_key=api_key)

    def evaluate_final_response(
        self,
        test_case: TestCase,
        agent_response: str,
        execution_time: float,
        total_tokens: int,
    ) -> EvaluationResult:
        """Judge one answer against its test case.

        Args:
            test_case: The case that was run.
            agent_response: What the agent finally replied.
            execution_time: Seconds the run took.
            total_tokens: Tokens the run consumed, if tracked.

        Returns:
            Scores plus the judge's explanation. If the judge's reply cannot be
            parsed the result scores 0 and says so, rather than guessing.
        """
        prompt = (
            "Evaluate this agent response for the given task:\n\n"
            f"Task: {test_case.description}\n"
            f"User Query: {test_case.user_query}\n"
            f"Agent Response: {agent_response}\n"
            f"Reference Answer: {test_case.reference_answer or 'No reference provided'}\n\n"
            "Rate the response on:\n"
            "1. Task completion: Did it fully answer the query?\n"
            "2. Format correctness: Is the format appropriate?\n"
            "3. Instruction following: Did it follow implicit instructions?\n\n"
            "Provide your evaluation with a brief explanation."
        )

        judge_response = self.llm_judge.invoke(prompt, response_format=JudgeEvaluation)

        try:
            evaluation = PydanticOutputParser(model_class=JudgeEvaluation).parse(
                judge_response
            )
        except Exception as error:
            evaluation = JudgeEvaluation(
                task_completed=False,
                format_correct=False,
                instructions_followed=False,
                explanation=f"Could not parse the judge's reply ({error}).",
            )

        scores = [
            evaluation.task_completed,
            evaluation.format_correct,
            evaluation.instructions_followed,
        ]

        return EvaluationResult(
            task_completion=TaskCompletionMetrics(
                task_completed=evaluation.task_completed,
                steps_taken=1,
                expected_steps=test_case.max_steps,
            ),
            quality_control=QualityControlMetrics(
                format_correct=evaluation.format_correct,
                instructions_followed=evaluation.instructions_followed,
            ),
            system_metrics=SystemMetrics(
                total_tokens=total_tokens,
                execution_time=execution_time,
                cost_estimate=total_tokens * COST_PER_TOKEN,
            ),
            overall_score=sum(scores) / len(scores),
            feedback=evaluation.explanation,
        )
