from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import anthropic
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import utils  # noqa: E402


class UtilsConfigTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.project_patch = patch.object(utils, "_PROJECT_ENV", {})
        self.project_patch.start()
        self.cc_patch = patch.object(utils, "_read_claude_code_settings", return_value={})
        self.cc_patch.start()

    def tearDown(self):
        self.cc_patch.stop()
        self.project_patch.stop()
        self.env_patch.stop()

    def test_no_provider_uses_process_env_then_claude_code_then_project_env(self):
        utils._PROJECT_ENV.update(
            {
                "ANTHROPIC_API_KEY": "project-key",
                "ANTHROPIC_BASE_URL": "https://project.invalid/anthropic",
                "LLM_MODEL": "project-model",
            }
        )
        with patch.object(
            utils,
            "_read_claude_code_settings",
            return_value={
                "ANTHROPIC_API_KEY": "claude-key",
                "ANTHROPIC_BASE_URL": "https://claude.invalid/anthropic",
                "ANTHROPIC_MODEL": "claude-model[1m]",
            },
        ):
            cfg = utils.get_llm_config()
        self.assertEqual(cfg["api_key"], "claude-key")
        self.assertEqual(cfg["base_url"], "https://claude.invalid/anthropic")
        self.assertEqual(cfg["model"], "claude-model")
        self.assertEqual(cfg["sources"]["api_key"], "claude_code")
        self.assertEqual(cfg["sources"]["base_url"], "claude_code")
        self.assertEqual(cfg["sources"]["model"], "claude_code")

        os.environ.update(
            {
                "ANTHROPIC_API_KEY": "environment-key",
                "ANTHROPIC_BASE_URL": "https://environment.invalid/anthropic",
                "LLM_MODEL": "environment-model",
            }
        )
        cfg = utils.get_llm_config()
        self.assertEqual(cfg["api_key"], "environment-key")
        self.assertEqual(cfg["base_url"], "https://environment.invalid/anthropic")
        self.assertEqual(cfg["model"], "environment-model")
        self.assertEqual(cfg["sources"]["api_key"], "environment")
        self.assertEqual(cfg["sources"]["base_url"], "environment")
        self.assertEqual(cfg["sources"]["model"], "environment")

    def test_selected_provider_does_not_fall_back_to_claude_code_credential(self):
        os.environ["LLM_PROVIDER"] = "minimax"
        with patch.object(
            utils,
            "_read_claude_code_settings",
            return_value={
                "ANTHROPIC_AUTH_TOKEN": "claude-token",
                "ANTHROPIC_BASE_URL": "https://claude.invalid/anthropic",
                "ANTHROPIC_MODEL": "claude-model",
            },
        ):
            cfg = utils.get_llm_config()
        self.assertEqual(cfg["provider_key"], "minimax")
        self.assertEqual(cfg["base_url"], "https://api.minimaxi.com/anthropic")
        self.assertEqual(cfg["model"], "MiniMax-M2.7")
        self.assertEqual(cfg["api_key"], "")
        self.assertEqual(cfg["sources"]["api_key"], "unset")

        utils._PROJECT_ENV["ANTHROPIC_AUTH_TOKEN"] = "project-token"
        cfg = utils.get_llm_config()
        self.assertEqual(cfg["api_key"], "project-token")
        self.assertEqual(cfg["credential_kind"], "auth_token")
        self.assertEqual(cfg["auth_mode"], "bearer")
        self.assertEqual(cfg["sources"]["api_key"], "project_env")

    def test_provider_preset_is_overridden_by_explicit_endpoint_and_model(self):
        os.environ.update(
            {
                "LLM_PROVIDER": "deepseek",
                "ANTHROPIC_API_KEY": "environment-key",
                "ANTHROPIC_BASE_URL": "https://proxy.invalid/anthropic",
                "LLM_MODEL": "custom-model[1m]",
            }
        )
        cfg = utils.get_llm_config()
        self.assertEqual(cfg["provider_key"], "deepseek")
        self.assertEqual(cfg["api_key"], "environment-key")
        self.assertEqual(cfg["base_url"], "https://proxy.invalid/anthropic")
        self.assertEqual(cfg["model"], "custom-model")
        self.assertEqual(cfg["sources"]["base_url"], "environment")
        self.assertEqual(cfg["sources"]["model"], "environment")

    def test_unknown_provider_is_a_configuration_error(self):
        os.environ["LLM_PROVIDER"] = "not-a-provider"
        with self.assertRaises(utils.LLMConfigError) as raised:
            utils.get_llm_config()
        self.assertIn("未知 LLM_PROVIDER", str(raised.exception))

    def test_401_is_non_retryable_authentication_error_without_fallback(self):
        request = httpx.Request("POST", "https://provider.invalid/anthropic")
        response = httpx.Response(401, request=request)
        secret = "synthetic-secret-token"
        error = anthropic.AuthenticationError(
            f"invalid credential {secret}",
            response=response,
            body={"error": {"message": "invalid credential"}},
        )
        info = utils.classify_llm_error(
            error,
            {
                "api_key": secret,
                "provider_key": "minimax",
                "model": "MiniMax-M2.7",
                "base_url": "https://provider.invalid/anthropic?token=hidden",
            },
        )
        self.assertEqual(info["category"], "authentication")
        self.assertEqual(info["error_type"], "authentication_error")
        self.assertEqual(info["status_code"], 401)
        self.assertFalse(info["retryable"])
        self.assertFalse(info["fallback_attempted"])
        self.assertNotIn(secret, info["message"])
        self.assertNotIn("?token=", info["message"])
        self.assertIn("未尝试切换其他 provider 或 key", info["message"])

    def test_redact_secrets_removes_bearer_and_key_values(self):
        secret = "synthetic-secret-token"
        text = utils.redact_secrets(
            f"Authorization: Bearer {secret}; key={secret}", secrets=[secret]
        )
        self.assertNotIn(secret, text)
        self.assertIn("<redacted>", text)

    def test_bearer_client_uses_selected_endpoint_and_does_not_use_sdk_env_fallback(self):
        import summarizer

        config = {
            "api_key": "synthetic-bearer-token",
            "base_url": "https://provider.invalid/anthropic",
            "model": "provider-model",
            "auth_mode": "bearer",
            "provider_key": "minimax",
            "provider_label": "MiniMax",
        }
        fake_http_client = Mock(name="http_client")
        with patch.object(summarizer, "_make_http_client", return_value=fake_http_client):
            with patch.object(summarizer, "_Anthropic") as constructor:
                summarizer.Anthropic(_llm_config=config)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["base_url"], config["base_url"])
        self.assertIsNone(kwargs["api_key"])
        self.assertEqual(
            kwargs["default_headers"]["Authorization"],
            "Bearer synthetic-bearer-token",
        )
        self.assertIs(kwargs["http_client"], fake_http_client)


if __name__ == "__main__":
    unittest.main()
