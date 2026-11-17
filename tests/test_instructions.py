from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from fastmcp.server.auth.auth import AccessToken, TokenVerifier

from statline.server import http
from statline.server.app import build_server
from statline.server.auth import build_auth
from statline.server.auth_config import AuthConfig
from statline.server.data import ServerData
from statline.server.http import build_http_app
from statline.server.instructions import (
    BASE_INSTRUCTIONS,
    LEAGUE_INSTRUCTIONS,
    MAX_INSTRUCTION_WORDS,
)


class RejectAll(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        return None


@pytest.mark.parametrize("text", [BASE_INSTRUCTIONS, LEAGUE_INSTRUCTIONS])
def test_instructions_fit_word_budget(text: str) -> None:
    assert len(text.split()) < MAX_INSTRUCTION_WORDS


def test_league_tools_only_in_league_instructions() -> None:
    assert "my_league" not in BASE_INSTRUCTIONS
    assert "free_agents" not in BASE_INSTRUCTIONS
    assert LEAGUE_INSTRUCTIONS.startswith(BASE_INSTRUCTIONS)
    assert "my_league" in LEAGUE_INSTRUCTIONS


def test_build_server_instructions_default_and_override(data: ServerData) -> None:
    assert build_server(data).instructions == BASE_INSTRUCTIONS
    assert build_server(data, instructions="custom").instructions == "custom"


def test_http_app_gives_each_server_its_instructions(
    data: ServerData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str] = {}

    def recording_build_server(*args: Any, **kwargs: Any) -> Any:
        server = build_server(*args, **kwargs)
        seen[server.name] = server.instructions or ""
        return server

    monkeypatch.setattr(http, "build_server", recording_build_server)
    cfg = AuthConfig(
        yahoo_client_id="cid",
        yahoo_client_secret="csecret",
        jwt_signing_key="j" * 40,
        storage_encryption_key=Fernet.generate_key(),
        base_url="https://statline.example",
        token_store="disk",
        token_dir=tmp_path / "oauth",
    )
    build_http_app(data, cfg, auth=build_auth(cfg, verifier=RejectAll()))
    assert seen == {"statline": BASE_INSTRUCTIONS, "statline-league": LEAGUE_INSTRUCTIONS}
