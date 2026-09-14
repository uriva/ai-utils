import json
import os
from pathlib import Path
import shlex
from typing import override

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.model_connection import ModelConnectionSpec, ResolvedModelConnection
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName

LOCAL_BINARY_PATH = Path("/tmp/opencode/ai-utils-agent")
REMOTE_BINARY_PATH = "/usr/local/bin/ai-utils-agent"
TRAJECTORY_REMOTE_PATH = "/logs/agent/trajectory.json"
OUTPUT_REMOTE_LOG = "/logs/agent/ai-utils-agent.txt"


class AiUtilsAgent(BaseInstalledAgent):
    """Harbor adapter for the uriva/ai-utils agent harness."""

    capabilities = AgentCapabilities(atif=False, resume=False)
    MODEL_CONNECTION = ModelConnectionSpec(passthrough=True)

    @classmethod
    def name(cls) -> str:
        return "ai-utils-agent"

    def get_version_command(self) -> str | None:
        return f"{REMOTE_BINARY_PATH} --help"

    def parse_version(self, stdout: str) -> str:
        return "0.3.19"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if not LOCAL_BINARY_PATH.exists():
            raise FileNotFoundError(
                f"Agent binary not found at {LOCAL_BINARY_PATH}. Run deno compile first."
            )

        # Upload the precompiled standalone agent binary
        await environment.upload_file(LOCAL_BINARY_PATH, REMOTE_BINARY_PATH)
        await self.exec_as_root(
            environment, command=f"chmod +x {REMOTE_BINARY_PATH}"
        )

        # Ensure logs directory exists
        await self.exec_as_root(
            environment, command="mkdir -p /logs/agent && chmod 777 /logs/agent"
        )

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)
        zai_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPU_API_KEY")
        if not zai_key:
            raise ValueError("ZAI_API_KEY or ZHIPU_API_KEY environment variable required")

        env = {
            "ZAI_API_KEY": zai_key,
            "ZHIPU_API_KEY": zai_key,
        }

        # Also pull in any extra env configured in Harbor
        if self._extra_env:
            env.update(self._extra_env)

        cmd = (
            f"{REMOTE_BINARY_PATH} --task {escaped_instruction} "
            f"--output {TRAJECTORY_REMOTE_PATH} --verbose "
            f"</dev/null > {OUTPUT_REMOTE_LOG} 2>&1"
        )

        await self.exec_as_agent(environment, command=cmd, env=env)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        local_trajectory = self.logs_dir / "agent" / "trajectory.json"
        if not local_trajectory.exists():
            local_trajectory = self.logs_dir / "trajectory.json"

        if local_trajectory.exists():
            try:
                data = json.loads(local_trajectory.read_text())
                usage = data.get("usage", {})
                context.n_input_tokens = usage.get("promptTokens", 0)
                context.n_output_tokens = usage.get("completionTokens", 0)
                context.n_cache_tokens = usage.get("cachedTokens", 0)
                context.cost_usd = data.get("costUsd", 0.0)
            except Exception as e:
                self.logger.warning(f"Could not parse trajectory stats: {e}")
