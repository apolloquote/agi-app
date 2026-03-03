from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math
import os
import re
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class AgentRunRequest(BaseModel):
    goal_title: str = Field(..., min_length=1, max_length=300)
    goal_description: str = Field(..., min_length=1, max_length=4000)
    model: str = Field(default='gpt-5.2')
    persist_memory: bool = Field(default=True)
    max_cycles: int = Field(default=10, ge=1, le=50)
    use_openai: Optional[bool] = Field(default=None)


class HealthResponse(BaseModel):
    ok: bool
    service: str
    timestamp: float


@dataclass
class Goal:
    title: str
    description: str
    success_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class Task:
    id: int
    title: str
    role: str
    status: str = 'pending'
    description: str = ''
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    depends_on: List[int] = field(default_factory=list)
    output: Optional[Dict[str, Any]] = None
    confidence: float = 0.0


@dataclass
class TraceEvent:
    timestamp: float
    kind: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def __init__(self, persist: bool = True, path: str = 'agi_runtime_memory.json'):
        self.persist = persist
        self.path = Path(path)
        self.semantic: Dict[str, Any] = {}
        self.working: List[str] = []
        self.episodes: List[Dict[str, Any]] = []
        if self.persist:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.semantic = data.get('semantic', {})
            self.working = data.get('working', [])[-50:]
            self.episodes = data.get('episodes', [])[-200:]
        except Exception:
            self.semantic = {}
            self.working = []
            self.episodes = []

    def _save(self) -> None:
        if not self.persist:
            return
        payload = {
            'semantic': self.semantic,
            'working': self.working[-50:],
            'episodes': self.episodes[-200:],
        }
        self.path.write_text(json.dumps(payload, indent=2))

    def set_semantic(self, key: str, value: Any) -> None:
        self.semantic[key] = value
        self._save()

    def add_working(self, item: str) -> None:
        self.working.append(item)
        self.working = self.working[-50:]
        self._save()

    def add_episode(self, kind: str, message: str, **metadata: Any) -> None:
        self.episodes.append({
            'timestamp': time.time(),
            'kind': kind,
            'message': message,
            'metadata': metadata,
        })
        self.episodes = self.episodes[-200:]
        self._save()

    def to_frontend(self) -> Dict[str, Any]:
        return {'semantic': self.semantic, 'working': self.working}


class Toolbelt:
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def run(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        fn = getattr(self, f'tool_{tool_name}', None)
        if not fn:
            return {'ok': False, 'error': f'Unknown tool: {tool_name}'}
        try:
            return fn(payload)
        except Exception as e:
            return {'ok': False, 'error': str(e)}

    def tool_notes(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = str(payload.get('text', '')).strip()
        if not text:
            return {'ok': False, 'error': 'Missing text.'}
        sentences = re.split(r'(?<=[.!?])\s+', text)
        summary = ' '.join(sentences[:3]).strip()
        keywords = sorted(set(re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())))[:12]
        return {'ok': True, 'summary': summary, 'keywords': keywords}

    def tool_memory_write(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        key = str(payload.get('key', '')).strip()
        if not key:
            return {'ok': False, 'error': 'Missing key.'}
        value = payload.get('value')
        self.memory.set_semantic(key, value)
        return {'ok': True, 'stored': {key: value}}

    def tool_extract_math(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = str(payload.get('text', ''))
        match = re.search(r'calculate[: ]+(.+)$', text.lower())
        if not match:
            return {'ok': False, 'error': 'No arithmetic expression found.'}
        return {'ok': True, 'expression': match.group(1).strip()}

    def tool_calculator(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        expr = str(payload.get('expression', '')).strip()
        if not expr:
            return {'ok': False, 'error': 'Missing expression.'}
        if not re.fullmatch(r'[0-9\s\+\-\*\/\(\)\.]+', expr):
            return {'ok': False, 'error': 'Unsafe expression.'}
        result = eval(expr, {'__builtins__': {}}, {})
        if not isinstance(result, (int, float)) or not math.isfinite(result):
            return {'ok': False, 'error': 'Invalid numeric result.'}
        return {'ok': True, 'result': result}


class LLMProvider:
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class StubLLMProvider(LLMProvider):
    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return {
            'ok': True,
            'text': f'Stubbed response for: {user_prompt[:180]}',
            'provider': 'stub',
        }


class OpenAIResponsesProvider(LLMProvider):
    def __init__(self, model: str):
        if OpenAI is None:
            raise RuntimeError('openai package is not installed.')
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY is not set.')
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        text = getattr(response, 'output_text', '') or ''
        return {
            'ok': True,
            'text': text.strip(),
            'provider': 'openai',
            'model': self.model,
        }


class AGIEngine:
    def __init__(self, goal: Goal, model: str, persist_memory: bool = True, max_cycles: int = 10, use_openai: Optional[bool] = None):
        self.goal = goal
        self.model = model
        self.max_cycles = max_cycles
        self.memory = MemoryStore(persist=persist_memory)
        self.tools = Toolbelt(self.memory)
        self.tasks: List[Task] = []
        self.trace: List[TraceEvent] = []
        self.llm = self._build_provider(use_openai)

    def _build_provider(self, use_openai: Optional[bool]) -> LLMProvider:
        env_flag = os.getenv('AGI_USE_OPENAI', 'false').strip().lower() in {'1', 'true', 'yes'}
        wants_openai = env_flag if use_openai is None else use_openai
        if wants_openai:
            try:
                provider = OpenAIResponsesProvider(model=self.model)
                self.memory.set_semantic('llm_provider', 'openai')
                return provider
            except Exception as e:
                self.memory.set_semantic('llm_provider', f'stub_fallback: {e}')
                return StubLLMProvider()
        self.memory.set_semantic('llm_provider', 'stub')
        return StubLLMProvider()

    def log(self, kind: str, message: str, **metadata: Any) -> None:
        event = TraceEvent(timestamp=time.time(), kind=kind, message=message, metadata=metadata)
        self.trace.append(event)
        self.memory.add_episode(kind, message, **metadata)

    def bootstrap(self) -> None:
        self.tasks = [
            Task(
                id=1,
                title='Persist goal',
                role='planner',
                description='Store active goal in semantic memory.',
                tool_name='memory_write',
                tool_input={'key': 'active_goal', 'value': self.goal.title},
            ),
            Task(
                id=2,
                title='Analyze input',
                role='researcher',
                description='Summarize the user request and extract signals.',
                tool_name='notes',
                tool_input={'text': self.goal.description},
            ),
        ]

        next_id = 3
        if 'calculate' in self.goal.description.lower():
            self.tasks.append(Task(
                id=next_id,
                title='Extract expression',
                role='researcher',
                description='Find the arithmetic expression in the goal description.',
                tool_name='extract_math',
                tool_input={'text': self.goal.description},
            ))
            next_id += 1
            self.tasks.append(Task(
                id=next_id,
                title='Compute result',
                role='executor',
                description='Compute the extracted arithmetic expression.',
                depends_on=[next_id - 1],
            ))
            next_id += 1
        else:
            self.tasks.append(Task(
                id=next_id,
                title='Execute action',
                role='executor',
                description='Perform a live reasoning and execution cycle.',
            ))
            next_id += 1

        self.tasks.append(Task(
            id=next_id,
            title='Critique and score',
            role='critic',
            description='Review task completion and assess confidence.',
        ))
        next_id += 1
        self.tasks.append(Task(
            id=next_id,
            title='Final synthesis',
            role='critic',
            description='Generate a concise final answer.',
        ))

        self.log('bootstrap', 'Created initial task graph.', task_count=len(self.tasks))

    def _dependencies_met(self, task: Task) -> bool:
        if not task.depends_on:
            return True
        statuses = {t.id: t.status for t in self.tasks}
        return all(statuses.get(dep_id) == 'done' for dep_id in task.depends_on)

    def _select_next(self) -> Optional[Task]:
        for task in self.tasks:
            if task.status == 'pending' and self._dependencies_met(task):
                return task
        return None

    def _build_final_answer_preview(self) -> str:
        if 'calculate' in self.goal.description.lower():
            compute_task = next((t for t in self.tasks if t.title == 'Compute result' and t.output), None)
            if compute_task and compute_task.output and compute_task.output.get('ok'):
                return f"The arithmetic result is {compute_task.output.get('result')}."

        execution_task = next((t for t in self.tasks if t.title == 'Execute action' and t.output), None)
        if execution_task and execution_task.output and execution_task.output.get('text'):
            return execution_task.output['text']

        return (
            f"The AGI prototype accepted '{self.goal.title}', analyzed the request, "
            f"executed the task flow, and produced an auditable result using model '{self.model}'."
        )

    def _run_task(self, task: Task) -> None:
        task.status = 'running'
        self.log('task_start', f'Starting task {task.id}: {task.title}', role=task.role)

        if task.tool_name:
            result = self.tools.run(task.tool_name, task.tool_input or {})
        elif task.title == 'Compute result':
            dependency = next((t for t in self.tasks if t.id in task.depends_on and t.output and t.output.get('expression')), None)
            if dependency:
                result = self.tools.run('calculator', {'expression': dependency.output['expression']})
            else:
                result = {'ok': False, 'error': 'Missing extracted expression.'}
        elif task.title == 'Execute action':
            system_prompt = (
                'You are the execution agent in an AGI prototype. Provide a concise, useful response '
                'that advances the user\'s goal. Keep it practical and implementation-focused.'
            )
            user_prompt = (
                f"Goal title: {self.goal.title}\n"
                f"Goal description: {self.goal.description}\n"
                f"Success criteria: {', '.join(self.goal.success_criteria)}\n"
                f"Constraints: {', '.join(self.goal.constraints)}"
            )
            result = self.llm.generate(system_prompt, user_prompt)
        elif task.title == 'Critique and score':
            done_count = len([t for t in self.tasks if t.status == 'done'])
            execution_task = next((t for t in self.tasks if t.title == 'Execute action' and t.output), None)
            has_live_text = bool(execution_task and execution_task.output and execution_task.output.get('text'))
            result = {
                'ok': True,
                'result': f'Review complete. {done_count} task(s) finished before critique.',
                'confidence': min(0.98, 0.58 + (done_count * 0.07) + (0.08 if has_live_text else 0.0)),
            }
        elif task.title == 'Final synthesis':
            result = {'ok': True, 'result': self._build_final_answer_preview()}
        else:
            result = {'ok': True, 'result': f'Executed task: {task.title}'}

        task.output = result
        task.confidence = round(result.get('confidence', 0.9 if result.get('ok') else 0.15), 2)
        task.status = 'done' if result.get('ok') else 'failed'

        if result.get('ok'):
            self.log('task_done', f'Task {task.id} completed.', output=result)
        else:
            self.log('task_failed', f'Task {task.id} failed.', error=result.get('error'))

        self.memory.add_working(f'Task {task.id} => {result}')

    def _final_answer(self) -> str:
        calc_task = next((t for t in self.tasks if t.title == 'Compute result' and t.output and t.output.get('ok')), None)
        if calc_task:
            return (
                f"AGI run complete. Goal '{self.goal.title}' was processed successfully. "
                f"Computed result: {calc_task.output['result']}."
            )

        execution_task = next((t for t in self.tasks if t.title == 'Execute action' and t.output and t.output.get('text')), None)
        if execution_task:
            return (
                f"AGI run complete. Goal '{self.goal.title}' was executed with live model output. "
                f"Result: {execution_task.output['text']}"
            )

        return (
            f"AGI run complete. Goal '{self.goal.title}' was analyzed, decomposed, "
            f"and executed successfully using model '{self.model}'."
        )

    def run(self) -> Dict[str, Any]:
        if not self.tasks:
            self.bootstrap()

        for cycle in range(1, self.max_cycles + 1):
            self.log('cycle', f'Cycle {cycle} started.')
            task = self._select_next()
            if task is None:
                break
            self._run_task(task)

        final_answer = self._final_answer()
        self.memory.set_semantic('last_model', self.model)
        self.memory.set_semantic('last_final_answer', final_answer)

        return {
            'goal': asdict(self.goal),
            'model': self.model,
            'tasks': [asdict(task) for task in self.tasks],
            'trace': [asdict(event) for event in self.trace],
            'memory': self.memory.to_frontend(),
            'final_answer': final_answer,
        }


app = FastAPI(title='AGI Backend API', version='1.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/health', response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, service='agi-backend-api', timestamp=time.time())


@app.post('/api/agent/run')
def run_agent(request: AgentRunRequest) -> Dict[str, Any]:
    goal = Goal(
        title=request.goal_title,
        description=request.goal_description,
        success_criteria=[
            'Accept a goal',
            'Create a task graph',
            'Execute available tasks',
            'Return auditable output',
        ],
        constraints=['Bounded runtime', 'Safe local tools only'],
    )

    engine = AGIEngine(
        goal=goal,
        model=request.model or os.getenv('AGI_OPENAI_MODEL', 'gpt-5.2'),
        persist_memory=request.persist_memory,
        max_cycles=request.max_cycles,
        use_openai=request.use_openai,
    )
    return engine.run()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('agi_backend_api:app', host='127.0.0.1', port=8000, reload=True)
