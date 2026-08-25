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
    vars: Dict[str, Any]

class Step(Generic[StateSchema]):
    def __init__(self, step_id: str, logic: Callable[[StateSchema], Dict]):
        '''
        In plain English: defines one stop in the pipeline - a name plus the work to do
        there.

        It also checks up front how many arguments that work expects, so it does not
        have to be worked out again on every single run.

        Output: nothing returned; the step is ready to be placed on the map.
        '''
        self.step_id = step_id
        self.logic = logic
        # Store the number of parameters the logic function expects
        self.logic_params_count = self._calculate_params_count()

    def __str__(self) -> str:
        '''
        In plain English: how a step prints, e.g. `Step('retrieve')`.

        Output: that text.
        '''
        return f"Step('{self.step_id}')"

    def __repr__(self) -> str:
        '''
        In plain English: same as above, used when a step appears inside a list.

        Output: that text.
        '''
        return self.__str__()

    def _calculate_params_count(self):
        '''
        In plain English: works out whether this step's work needs one input or two.

        Some steps only need the running notes; others also need shared resources. This
        lets both be written naturally, without every step having to accept an argument
        it will not use.

        Output: the number of arguments, remembered and used on every run.
        '''
        """Calculate the number of parameters excluding 'self' for bound methods"""
        if inspect.ismethod(self.logic):
            # For bound methods, subtract 1 to exclude 'self'
            return self.logic.__func__.__code__.co_argcount - 1
        else:
            # For regular functions
            return self.logic.__code__.co_argcount

    def run(self, state: StateSchema, state_schema: Type[StateSchema], resource: Resource=None) -> StateSchema:
        # Call logic function with appropriate number of arguments
        '''
        In plain English: does this step's work and folds the result into the running
        notes.

        Two things worth knowing. It passes either one argument or two, depending on what
        the step asked for. And it merges rather than replaces: a step returns only the
        fields it changed, and everything else carries on untouched. That is why every
        step in this project can return a small dictionary and ignore the rest of the
        state.

        Output: the updated notes, passed on to whichever step comes next.
        '''
        if self.logic_params_count == 1:
            result = self.logic(state)
        elif self.logic_params_count == 2:
            result = self.logic(state, resource)
        else:
            raise ValueError(
                f"Step '{self.step_id}' logic function must accept either 1 argument (state) "
                f"or 2 arguments (state, resource). Found {self.logic_params_count} arguments."
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
        '''
        In plain English: the marker for where a run begins. It does no work itself.

        Output: nothing returned.
        '''
        super().__init__("__entry__", lambda x: {})


class Termination(Step[StateSchema]):
    """Special step that marks the end of the workflow.
    Users should connect their final business logic step(s) to this step."""
    def __init__(self):
        '''
        In plain English: the marker for where a run ends. It does no work itself.

        Output: nothing returned.
        '''
        super().__init__("__termination__", lambda x: {})


@dataclass
class Transition(Generic[StateSchema]):
    source: str
    targets: List[str]
    condition: Optional[Callable[[StateSchema], Union[str, List[str], Step[StateSchema], List[Step[StateSchema]]]]] = None

    def __str__(self) -> str:
        '''
        In plain English: how a connection prints, showing what leads where.

        Output: text like `Transition('evaluate' -> ['answer', 'recall'])`.
        '''
        return f"Transition('{self.source}' -> {self.targets})"

    def __repr__(self) -> str:
        '''
        In plain English: same as above, used when a connection appears inside a list.

        Output: that text.
        '''
        return self.__str__()

    def resolve(self, state: StateSchema) -> List[str]:
        '''
        In plain English: works out which step actually comes next.

        If the connection has a condition attached, the condition is asked, and it may
        answer with a step or just a name. If there is no condition, the destination was
        fixed when the map was drawn. This is the mechanism the whole project's branching
        rests on.

        Output: the name of the next step to run.
        '''
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
    state_data: StateSchema
    state_schema: Type[StateSchema]
    step_id: str

    def __str__(self) -> str:
        '''
        In plain English: how one recorded moment prints - when it happened, at which
        step, and the notes at that time.

        Output: that text.
        '''
        return f"Snapshot('{self.snapshot_id}') @ [{self.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')}]: {self.step_id}.State({self.state_data})"

    def __repr__(self) -> str:
        '''
        In plain English: same as above, used inside a list.

        Output: that text.
        '''
        return self.__str__()

    @classmethod
    def create(cls, state_data: StateSchema, state_schema: Type[StateSchema],
               step_id:str) -> 'Snapshot[StateSchema]':
        '''
        In plain English: records the state of the notes at one moment, stamped with the
        time and the step.

        Output: the recorded moment. Collected during a run, and it is this record that
        later reveals which route a question took.
        '''
        return cls(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            state_data=state_data,
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
        '''
        In plain English: how a run prints - just its identifier.

        Output: that text.
        '''
        return f"Run('{self.run_id}')"

    def __repr__(self) -> str:
        '''
        In plain English: same as above, used inside a list.

        Output: that text.
        '''
        return self.__str__()

    @classmethod
    def create(cls) -> 'Run[StateSchema]':
        '''
        In plain English: starts a new run and stamps the time it began.

        Output: the fresh run, ready to collect recorded moments.
        '''
        return cls(
            run_id=str(uuid.uuid4()),
            start_timestamp=datetime.now()
        )

    @property
    def metadata(self) -> Dict:
        '''
        In plain English: a short summary of the run - its identifier, when it started
        and finished, how many moments were recorded.

        Output: those details as a dictionary, for logging.
        '''
        return {
            "run_id": self.run_id,
            "start_timestamp": self.start_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "end_timestamp": self.end_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "snapshot_counts": len(self.snapshots)
        }

    def add_snapshot(self, snapshot: Snapshot[StateSchema]):
        '''
        In plain English: files one recorded moment into this run.

        Output: nothing returned. Called automatically after every step.
        '''
        """Add a new snapshot to this run"""
        self.snapshots.append(snapshot)

    def complete(self):
        '''
        In plain English: stamps the time the run finished.

        Output: nothing returned. The gap between the two stamps is what the evaluation
        reports as execution time.
        '''
        """Mark this run as complete"""
        self.end_timestamp = datetime.now()

    def get_final_state(self) -> Optional[StateSchema]:
        '''
        In plain English: the notes as they stood when the run ended.

        Output: the final state - which is where the answer lives. This is the method
        almost every caller in the project uses to get a result out of a run.
        '''
        """Get the final state of this run"""
        if not self.snapshots:
            return None
        return self.snapshots[-1].state_data


class StateMachine(Generic[StateSchema]):
    def __init__(self, state_schema: Type[StateSchema]):
        '''
        In plain English: creates an empty map, told in advance what shape the notes
        will have.

        Output: nothing returned. Steps and connections are added next.
        '''
        self.state_schema = state_schema
        self.steps: Dict[str, Step[StateSchema]] = {}
        self.transitions: Dict[str, List[Transition[StateSchema]]] = {}

    def __str__(self) -> str:
        '''
        In plain English: how a machine prints - the field names its notes carry.

        Output: that text.
        '''
        schema_keys = list(get_type_hints(self.state_schema).keys())
        return f"StateMachine(schema={schema_keys})"

    def __repr__(self) -> str:
        '''
        In plain English: same as above, used inside a list.

        Output: that text.
        '''
        return self.__str__()

    def add_steps(self, steps: List[Step[StateSchema]]):
        '''
        In plain English: puts steps on the map, so they can be referred to by name.

        Output: nothing returned. Adding a step does not connect it to anything - that
        is the next method's job.
        '''
        """Add steps to the workflow"""
        for step in steps:
            self.steps[step.step_id] = step

    def connect(
        self,
        source: Union[Step[StateSchema], str],
        targets: Union[Step[StateSchema], str, List[Union[Step[StateSchema], str]]],
        condition: Optional[Callable[[StateSchema], Union[str, List[str]]]] = None
    ):
        '''
        In plain English: draws an arrow from one step to another, optionally with a
        condition on it.

        Without a condition the route is fixed. With one, the condition decides at the
        moment of travel. Every branch in this project is one of these.

        Output: nothing returned; the arrow is added to the map.
        '''
        src_id = source.step_id if isinstance(source, Step) else source
        target_list = targets if isinstance(targets, list) else [targets]
        target_ids = [t.step_id if isinstance(t, Step) else t for t in target_list]
        transition = Transition[StateSchema](source=src_id, targets=target_ids, condition=condition)
        if src_id not in self.transitions:
            self.transitions[src_id] = []
        self.transitions[src_id].append(transition)

    def run(self, state: StateSchema, resource: Resource = None):
        # Validate that state has at least one field from the schema
        '''
        In plain English: walks the map from start to finish, doing the work at each
        stop.

        For each step it does the work, records a snapshot, then asks which arrow to
        follow. It stops at the end marker. Two situations are treated as programming
        errors rather than handled: a step with nowhere to go, and a condition that
        returns more than one destination, since running two branches at once is not
        supported.

        Output: the completed run - the final notes plus the full record of which steps
        were visited and in what order.
        '''
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
