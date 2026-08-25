from typing import TypedDict, List
import logging

from .state_machine import StateMachine, Step, EntryPoint, Termination, Run, Resource
from .llm import LLM
from .messages import BaseMessage, UserMessage, SystemMessage
from .vector_db import VectorStore


logging.getLogger('pdfminer').setLevel(logging.ERROR)


class RAGState(TypedDict):
    """
    Type definition for the state object passed through the RAG pipeline.
    """
    messages: List[BaseMessage]
    question: str
    documents: List[str]
    distances: List[float]
    answer: str

class RAG:
    """
    Retrieval-Augmented Generation (RAG) system implementation.
    
    This class orchestrates the complete RAG pipeline using a state machine approach:
    1. Retrieve: Find relevant documents using vector similarity search
    2. Augment: Combine retrieved context with the user's question
    3. Generate: Use an LLM to produce an answer based on the augmented prompt
    
    The RAG pattern enhances LLM responses by providing relevant external knowledge,
    reducing hallucinations and improving factual accuracy.
    """
    def __init__(self, llm: LLM, vector_store: VectorStore):
        '''
        In plain English: sets up the classic three-step retrieval pipeline - find, add
        context, answer.

        Output: nothing returned; ready to be used through `invoke`.
        '''
        self.workflow = self._create_state_machine()
        self.resource = Resource(
            services = {
                "llm": llm,
                "vector_store": vector_store,
            }
        )

    def _retrieve(self, state:RAGState, resource:Resource) -> RAGState:
        '''
        In plain English: step one - find the documents closest in meaning to the
        question.

        Output: the documents and their distances, put into the shared notes for the
        next step.
        '''
        question = state["question"]
        vector_store:VectorStore = resource.services.get("vector_store")
        results = vector_store.query(query_texts=[question])

        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        return {"documents": documents, "distances": distances}

    def _augment(self, state:RAGState) -> RAGState:
        '''
        In plain English: step two - build the prompt, with the found documents pasted
        in as context.

        This is the "augmented" part of retrieval-augmented generation: the model is not
        asked from memory, it is asked to read.

        Output: the finished conversation, ready to send.
        '''
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join(documents)

        messages = [
            SystemMessage(content="You are an assistant for question-answering tasks."),
            UserMessage(
                content=(
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    f"\n# Question: \n-> {question} "
                    f"\n# Context: \n-> {context} "
                    "\n# Answer: "
                )
            )
        ]

        return {"messages": messages}

    def _generate(self, state:RAGState, resource:Resource) -> RAGState:
        '''
        In plain English: step three - send the prompt and get the written answer.

        Output: the answer text, plus the conversation including the reply, in case a
        caller wants to see exactly what was asked.
        '''
        llm:LLM = resource.services.get("llm")
        ai_message = llm.invoke(state["messages"])
        return {
            "answer": ai_message.content, 
            "messages": state["messages"] + [ai_message],
        }

    def _create_state_machine(self) -> StateMachine[RAGState]:
        '''
        In plain English: wires the three steps into a fixed order, with no branching.

        Output: the assembled pipeline, used by `invoke`.
        '''
        machine = StateMachine[RAGState](RAGState)

        # Create steps
        entry = EntryPoint[RAGState]()
        retrieve = Step[RAGState]("retrieve", self._retrieve)
        augment = Step[RAGState]("augment", self._augment)
        generate = Step[RAGState]("generate", self._generate)
        termination = Termination[RAGState]()

        machine.add_steps([entry, retrieve, augment, generate, termination])
        machine.connect(entry, retrieve)
        machine.connect(retrieve, augment)
        machine.connect(augment, generate)
        machine.connect(generate, termination)

        return machine

    def invoke(self, query: str) -> Run:
        '''
        In plain English: run the three steps for one question.

        Not used by the research agent, which retrieves and answers in separate nodes
        of its own state machine; this is the self-contained RAG pipeline the framework
        offers on its own.

        Output: a run object holding the documents, the distances and the generated
        answer.
        '''
        """
        Execute the complete RAG pipeline for a given query.
        
        This is the main entry point for the RAG system. It initializes the
        pipeline state with the user's question and executes the complete
        retrieve-augment-generate workflow.
        
        Args:
            query (str): The user's question or search query
            
        Returns:
            Run: Execution object containing the final state and pipeline results
            
        Example:
            >>> rag = RAG(llm, vector_store)
            >>> result = rag.invoke("What is machine learning?")
            >>> answer = result.get_final_state()["answer"]
        """
        
        initial_state: RAGState = {
            "question": query,
        }
        run_object = self.workflow.run(
            state = initial_state, 
            resource = self.resource,
        )
        return run_object
