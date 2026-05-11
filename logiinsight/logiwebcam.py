"""
Copyright (c) 2026 Logitech Europe S.A. (LOGITECH) All Rights Reserved. This program is a trade
secret of LOGITECH, and it is not to be reproduced, published, disclosed to others, copied,
adapted, distributed or displayed without the prior authorization of LOGITECH. Licensee agrees
to attach or embed this notice on all copies of the program, including partial copies or
modified versions thereof.

LogiInsight Python API — Host-side client for controlling AI models and
reading real-time tracking data from the LogiInsight device over USB NCM.

Requirements:
    pip install paramiko
"""

import json
import logging
from typing import Dict, Optional

import paramiko

logger = logging.getLogger(__name__)

# Device file paths
_AI_CONFIG_PATH = "/files/logi_conf/logiwebcam_ai.ini"
_AI_OUTPUT_PATH = "/files/logi_conf/logiwebcam_ai_output.json"
_INI_SECTION = "logiwebcam_ai"


class LogiWebcamConnectionError(Exception):
    """Raised when the SSH connection to the device fails."""

    pass


class LogiWebcamDataError(Exception):
    """Raised when AI output data cannot be read or parsed."""

    pass


class LogiWebcamClient:
    """Client for controlling LogiInsight AI models over USB NCM (SSH).

    Manages an SSH connection to the device and provides methods to toggle
    AI inference models and read real-time face/hand tracking data.

    Example:
        >>> with LogiWebcamClient() as cam:
        ...     cam.set_ai_config(enable_face_detection=True, enable_face_mesh=True)
        ...     data = cam.get_latest_ai_data()
        ...     print(data["faces"])
    """

    def __init__(
        self,
        ip: str = "192.168.1.10",
        username: str = "makentu",
        password: str = "MakeNTU@123",
        port: int = 22,
        timeout: int = 10,
    ) -> None:
        """Initialize the client with connection parameters.

        Args:
            ip: Device IP address (default: 192.168.1.10 via USB NCM).
            username: SSH username on the device.
            password: SSH password on the device.
            port: SSH port number.
            timeout: Connection timeout in seconds.
        """
        self._ip = ip
        self._username = username
        self._password = password
        self._port = port
        self._timeout = timeout
        self._ssh: Optional[paramiko.SSHClient] = None

    def connect(self) -> None:
        """Establish an SSH connection to the device.

        Raises:
            LogiWebcamConnectionError: If the connection fails.
        """
        if self._ssh is not None:
            return

        try:
            self._ssh = paramiko.SSHClient()
            self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self._ssh.connect(
                hostname=self._ip,
                port=self._port,
                username=self._username,
                password=self._password,
                look_for_keys=False,
                timeout=self._timeout,
            )
            logger.info("Connected to LogiInsight at %s:%d", self._ip, self._port)
        except Exception as e:
            self._ssh = None
            raise LogiWebcamConnectionError(
                f"Failed to connect to {self._ip}:{self._port}: {e}"
            ) from e

    def disconnect(self) -> None:
        """Close the SSH connection."""
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None
            logger.info("Disconnected from LogiInsight")

    def __enter__(self) -> "LogiWebcamClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def _exec(self, command: str, timeout: int = 10) -> str:
        """Execute a command on the device and return stdout.

        Args:
            command: Shell command to execute.
            timeout: Command timeout in seconds.

        Returns:
            The stdout output as a string.

        Raises:
            LogiWebcamConnectionError: If not connected.
        """
        if self._ssh is None:
            raise LogiWebcamConnectionError("Not connected. Call connect() first.")

        _, stdout, stderr = self._ssh.exec_command(command, timeout=timeout)
        return stdout.read().decode().strip()

    def set_ai_config(
        self,
        enable_face_detection: bool = True,
        enable_face_mesh: bool = False,
        enable_hands: bool = False,
    ) -> bool:
        """Update the AI model configuration on the device.

        Writes the configuration to the device's INI file, which the firmware
        picks up via inotify within ~200ms.

        Args:
            enable_face_detection: Enable face bounding box detection.
            enable_face_mesh: Enable 478-point face landmark mesh.
                Requires enable_face_detection=True.
            enable_hands: Enable hand landmark tracking.

        Returns:
            True if the configuration was successfully written.

        Raises:
            LogiWebcamConnectionError: If not connected.
        """
        ini_content = (
            f"[{_INI_SECTION}]\n"
            f"enable_face_detection={int(enable_face_detection)}\n"
            f"enable_face_mesh={int(enable_face_mesh)}\n"
            f"enable_hands={int(enable_hands)}\n"
        )

        try:
            self._exec(f"echo '{ini_content}' > {_AI_CONFIG_PATH}")
            logger.info(
                "AI config set: face_det=%s, face_mesh=%s, hands=%s",
                enable_face_detection,
                enable_face_mesh,
                enable_hands,
            )
            return True
        except Exception as e:
            logger.error("Failed to set AI config: %s", e)
            return False

    def get_latest_ai_data(self) -> Optional[Dict]:
        """Fetch the latest AI tracking data from the device.

        Reads and parses the JSON output file written by the firmware.
        Returns None if the file is empty, missing, or malformed.

        Returns:
            A dictionary with keys 'timestamp_ms', 'faces', 'hands',
            or None if data is unavailable.

        Raises:
            LogiWebcamConnectionError: If not connected.
        """
        try:
            raw = self._exec(f"cat {_AI_OUTPUT_PATH}")
            if not raw:
                return None
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Malformed JSON from device, skipping frame")
            return None
        except Exception as e:
            logger.warning("Failed to read AI data: %s", e)
            return None

    @property
    def is_connected(self) -> bool:
        """Check if the SSH connection is active."""
        if self._ssh is None:
            return False
        transport = self._ssh.get_transport()
        return transport is not None and transport.is_active()
