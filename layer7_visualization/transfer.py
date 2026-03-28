"""
Exam Room Transfer
====================
Transfers the generated PDF report to the exam room for the
veterinarian. Supports SMB file share (clinic LAN) and local
copy (development fallback).
"""

from __future__ import annotations

import os
import shutil

from cap.common.logging_setup import get_logger

logger = get_logger("visualization.transfer")


class ExamRoomTransfer:
    """
    Transfers PDF reports to the exam room.
    """

    def __init__(self, config: object) -> None:
        if hasattr(config, "transfer"):
            self._protocol = config.transfer.protocol
            self._target_path = config.transfer.target_path
            self._fallback_path = config.transfer.fallback_local_path
        else:
            transfer = config.get("transfer", {})
            self._protocol = transfer.get("protocol", "local_copy")
            self._target_path = transfer.get("target_path")
            self._fallback_path = transfer.get("fallback_local_path", "./data/exports/reports/")

        logger.info(
            "ExamRoomTransfer initialized: protocol=%s, target=%s",
            self._protocol, self._target_path or "(not configured)",
        )

    def transfer(self, pdf_path: str, filename: str = None) -> str:
        """
        Transfer a PDF report to the exam room.

        Parameters
        ----------
        pdf_path : str
            Path to the source PDF file.
        filename : str, optional
            Output filename. If None, uses the source filename.

        Returns
        -------
        str
            Path where the file was copied.

        Raises
        ------
        FileNotFoundError
            If the source PDF doesn't exist.
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if filename is None:
            filename = os.path.basename(pdf_path)

        if self._protocol == "smb_share" and self._target_path:
            return self._transfer_smb(pdf_path, filename)
        else:
            return self._transfer_local(pdf_path, filename)

    def _transfer_smb(self, pdf_path: str, filename: str) -> str:
        """Transfer via SMB/CIFS file share (network copy)."""
        try:
            target = os.path.join(self._target_path, filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(pdf_path, target)
            logger.info("Report transferred via SMB: %s → %s", pdf_path, target)
            return target
        except (OSError, PermissionError) as e:
            logger.warning(
                "SMB transfer failed (%s), falling back to local copy", e
            )
            return self._transfer_local(pdf_path, filename)

    def _transfer_local(self, pdf_path: str, filename: str) -> str:
        """Fallback: copy to local directory."""
        os.makedirs(self._fallback_path, exist_ok=True)
        target = os.path.join(self._fallback_path, filename)
        shutil.copy2(pdf_path, target)
        logger.info("Report copied locally: %s → %s", pdf_path, target)
        return target

    @property
    def is_network_configured(self) -> bool:
        """Whether a network target path is configured."""
        return bool(self._target_path)
