from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic, cast, Type, TypedDict, get_type_hints
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import copy
import inspect
import logging


logger = logging.getLogger(__name__)

StateSchema = TypeVar("StateSchema")

@dataclass
class Resource:
    """Long-lived objects a step may need, keyed by name."""

    services: Dict[str, Any]

class Step(Generic[StateSchema]):
    def __init__(self, step_id: str, action: Callable[[StateSchema], Dict]):
        self.step_id = step_id
        self.action = action
        # Store the number of parameters the action expects
        self.action_params_count = self._calculate_params_count()

    def __str__(self) -> str:
        return f"Step('{self.step_id}')"

    def __repr__(self) -> str:
        return self.__str__()

    def _calculate_params_count(self):
        """Calculate the number of parameters excluding 'self' for bound methods"""
        if inspect.ismethod(self.action):
            # For bound methods, subtract 1 to exclude 'self'
            return self.action.__func__.__code__.co_argcount - 1
        else:
            # For regular functions
            return self.action.__code__.co_argcount

    def run(self, state: StateSchema, state_schema: Type[StateSchema], resource: Resource=None) -> StateSchema:
        """Run the step's action and merge its result into the state.

        The action is called with `(state)` or `(state, resource)` according to
        its signature; only keys declared in `state_schema` are merged.

        Raises:
            ValueError: If the action accepts neither 1 nor 2 arguments.
        """
        if self.action_params_count == 1:
            result = self.action(state)
        elif self.action_params_count == 2:
            result = self.action(state, resource)
        else:
            raise ValueError(
                f"Step '{self.step_id}' action must accept either 1 argument (state) "
                f"or 2 arguments (state, resource). Found {self.action_params_count} arguments."
            ) 
        # Get expected fields from the TypedDict
        expected_fields = get_type_hints(state_schema)
        
        # Create new state with all fields from state_schema
        # Only copy fields that are defined in state_schema
        updated = {**state}
        for field, value in result.items():
            if field in expected_fields:
                updated[field] = value
        
        return cast(StateSchema, updated)


class EntryPoint(Step[StateSchema]):
    """Special step that marks the beginning of the workflow.
    Users should connect this step to their first business logic step."""
    def __init__(self):
        super().__init__("__entry__", lambda x: {})


class Termination(Step[StateSchema]):
    """Special step that marks the end of the workflow.
    Users should connect their final business logic step(s) to this step."""
    def __init__(self):
        super().__init__("__termination__", lambda x: {})


@dataclass
class Transition(Generic[StateSchema]):
    source: str
    targets: List[str]
    condition: Optional[Callable[[StateSchema], Union[str, List[str], Step[StateSchema], List[Step[StateSchema]]]]] = None

    def __str__(self) -> str:
        return f"Transition('{self.source}' -> {self.targets})"

    def __repr__(self) -> str:
        return self.__str__()

    def resolve(self, state: StateSchema) -> List[str]:
        """Return the ids of the next steps, applying the condition if one is set."""
        if self.condition:
            result = self.condition(state)
            if isinstance(result, Step):
                return [result.step_id]
            elif isinstance(result, list) and all(isinstance(x, Step) for x in result):
                return [step.step_id for step in result]
            elif isinstance(result, str):
                return [result]
            return result
        return self.targets


@dataclass
class Snapshot(Generic[StateSchema]):
    """Represents a single state snapshot in time"""
    snapshot_id: str
    timestamp: datetime
    state: StateSchema
    state_schema: Type[StateSchema]
    step_id: str

    def __str__(self) -> str:
        return f"Snapshot('{self.snapshot_id}') @ [{self.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')}]: {self.step_id}.State({self.state})"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def create(cls, state: StateSchema, state_schema: Type[StateSchema],
               step_id:str) -> 'Snapshot[StateSchema]':
        """Capture the state at one step, stamped with the current time."""
        return cls(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            state=state,
            state_schema=state_schema,
            step_id=step_id,
        )


@dataclass
class Run(Generic[StateSchema]):
    """Represents a single execution run of the state machine"""
    run_id: str
    start_timestamp: datetime
    snapshots: List[Snapshot[StateSchema]] = field(default_factory=list)
    end_timestamp: Optional[datetime] = None

    def __str__(self) -> str:
        return f"Run('{self.run_id}')"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def create(cls) -> 'Run[StateSchema]':
        """Start a new run, stamped with the current time."""
        return cls(
            run_id=str(uuid.uuid4()),
            start_timestamp=datetime.now()
        )

    @property
    def metadata(self) -> Dict:
        """Return the run id, its start and end times, and the snapshot count."""
        return {
            "run_id": self.run_id,
            "start_timestamp": self.start_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "end_timestamp": self.end_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "snapshot_counts": len(self.snapshots)
        }

    def add_snapshot(self, snapshot: Snapshot[StateSchema]):
        """Add a new snapshot to this run"""
        self.snapshots.append(snapshot)

    def complete(self):
        """Mark this run as complete"""
        self.end_timestamp = datetime.now()

    def get_final_state(self) -> Optional[StateSchema]:
        """Get the final state of this run"""
        if not self.snapshots:
            return None
        return self.snapshots[-1].state


class StateMachine(Generic[StateSchema]):
    def __init__(self, state_schema: Type[StateSchema]):
        self.state_schema = state_schema
        self.steps: Dict[str, Step[StateSchema]] = {}
        self.transitions: Dict[str, List[Transition[StateSchema]]] = {}

    def __str__(self) -> str:
        schema_keys = list(get_type_hints(self.state_schema).keys())
        return f"StateMachine(schema={schema_keys})"

    def __repr__(self) -> str:
        return self.__str__()

    def add_steps(self, steps: List[Step[StateSchema]]):
        """Add steps to the workflow"""
        for step in steps:
            self.steps[step.step_id] = step

    def connect(
        self,
        source: Union[Step[StateSchema], str],
        targets: Union[Step[StateSchema], str, List[Union[Step[StateSchema], str]]],
        condition: Optional[Callable[[StateSchema], Union[str, List[str]]]] = None
    ):
        """Connect a source step to one or more target steps.

        Args:
            source: Step or step id the transition leaves from.
            targets: Step(s) or id(s) it may lead to.
            condition: Called with the state to choose among `targets`.
                Without one, the transition is unconditional.
        """
        src_id = source.step_id if isinstance(source, Step) else source
        target_list = targets if isinstance(targets, list) else [targets]
        target_ids = [t.step_id if isinstance(t, Step) else t for t in target_list]
        transition = Transition[StateSchema](source=src_id, targets=target_ids, condition=condition)
        if src_id not in self.transitions:
            self.transitions[src_id] = []
        self.transitions[src_id].append(transition)

    def run(self, state: StateSchema, resource: Resource = None):
        """Execute the machine from its entry point until termination.

        Args:
            state: Initial state; must share at least one key with the schema.
            resource: Passed to steps whose action takes two arguments.

        Returns:
            The completed `Run`, holding one snapshot per step visited.

        Raises:
            ValueError: If the initial state shares no key with the schema.
            NotImplementedError: If a condition resolves to more than one target.
        """
        expected_fields = get_type_hints(self.state_schema)
        state_fields = set(state.keys())
        common_fields = state_fields.intersection(expected_fields)
        
        if not common_fields:
            raise ValueError(f"Initial state must have at least one field from the schema. Expected fields: {list(expected_fields.keys())}")

        entry_points = [s for s in self.steps.values() if isinstance(s, EntryPoint)]
        if not entry_points:
            raise Exception("No EntryPoint step found in workflow")
        if len(entry_points) > 1:
            raise Exception("Multiple EntryPoint steps found in workflow")
        
        # Create a new run for this execution
        current_run = Run.create()
        
        current_step_id = entry_points[0].step_id        

        while current_step_id:
            step = self.steps[current_step_id]
            if isinstance(step, Termination):
                logger.debug("terminating: %s", current_step_id)
                break
            
            # Replace state entirely
            state = step.run(state, self.state_schema, resource)  

            if isinstance(step, EntryPoint):
                logger.debug("starting: %s", current_step_id)
            else:
                logger.debug("step: %s", current_step_id)

            # Create and add snapshot to the current run
            snapshot = Snapshot.create(copy.deepcopy(state), self.state_schema, current_step_id)
            current_run.add_snapshot(snapshot)

            transitions = self.transitions.get(current_step_id, [])
            next_steps: List[str] = []

            for t in transitions:
                next_steps += t.resolve(state)

            if not next_steps:
                raise Exception(f"[StateMachine] No transitions found from step: {current_step_id}")

            if len(next_steps) > 1:
                raise NotImplementedError("Parallel execution not implemented yet.")

            current_step_id = next_steps[0]

        current_run.complete()
        return current_run
