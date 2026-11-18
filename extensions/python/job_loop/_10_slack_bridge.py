"""Keep the Slack chat bridge running.

Runs on every job_loop tick (~60 s). If chat_bridge.auto_start is true and both
tokens are configured, starts the bridge when it is not running (first boot,
after a WebUI restart, or after a crash).
"""

import logging

from helpers.extension import Extension

logger = logging.getLogger("slack_chat_bridge")


class SlackBridgeAutoStart(Extension):
    async def execute(self, **kwargs):
        try:
            from usr.plugins.slack.helpers.slack_client import get_slack_config
            from usr.plugins.slack.helpers.slack_bot import get_bot_status, start_chat_bridge

            config = get_slack_config() or {}
            if not config.get("chat_bridge", {}).get("auto_start", False):
                return
            bot_token = config.get("bot", {}).get("token", "")
            app_token = config.get("bot", {}).get("app_token", "")
            if not bot_token or not app_token:
                return

            status = get_bot_status()
            if status.get("running"):
                return

            logger.info(f"Slack bridge not running ({status.get('status')}); auto-starting")
            await start_chat_bridge(bot_token, app_token)
            logger.info("Slack chat bridge auto-started")
        except Exception as e:
            logger.warning(f"Slack chat bridge auto-start failed: {type(e).__name__}: {e}")
