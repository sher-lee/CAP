
"""
CAP Main Window
================
The root application window. Contains a QStackedWidget that
manages all 7 screens. Handles navigation, window title,
and global keyboard shortcuts (e.g. emergency stop).

Usage:
    from cap.layer6_ui.main_window import launch_ui
    launch_ui(app_context)
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QMessageBox,
    QStatusBar, QToolBar,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QKeySequence

from cap.common.logging_setup import get_logger
from cap.layer6_ui.signals import NavigationSignals

if TYPE_CHECKING:
    from cap.app import AppContext

logger = get_logger("ui.main")

# Screen name constants
SCREEN_SESSION_START = "session_start"
SCREEN_SCAN_REGION = "scan_region"
SCREEN_SCAN_CONTROL = "scan_control"
SCREEN_RESULTS = "results_dashboard"
SCREEN_REVIEW = "review_confirm"
SCREEN_HISTORY = "patient_history"
SCREEN_SETTINGS = "settings"

SCREEN_ORDER = [
    SCREEN_SESSION_START,
    SCREEN_SCAN_REGION,
    SCREEN_SCAN_CONTROL,
    SCREEN_RESULTS,
    SCREEN_REVIEW,
    SCREEN_HISTORY,
    SCREEN_SETTINGS,
]


class MainWindow(QMainWindow):
    """
    Root window containing all screens in a QStackedWidget.
    """

    def __init__(self, app_context: AppContext) -> None:
        super().__init__()

        self._ctx = app_context
        self._nav_signals = NavigationSignals()

        # Window setup
        self.setWindowTitle("CAP — Cytology Analysis Project")
        self.setMinimumSize(1024, 700)
        self.resize(1280, 800)

        # Central stacked widget
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Create and register all screens
        self._screens: dict[str, object] = {}
        self._create_screens()

        # Navigation toolbar
        self._create_toolbar()

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — select a technician to begin")

        # Connect navigation signals
        self._nav_signals.go_to_screen.connect(self.navigate_to)
        self._nav_signals.go_back.connect(self._go_back)

        # Global shortcuts
        self._setup_shortcuts()

        # Start on session screen
        self.navigate_to(SCREEN_SESSION_START)

        logger.info("MainWindow initialized with %d screens", len(self._screens))

    def _create_screens(self) -> None:
        """Instantiate all screen widgets and add to the stack."""
        from cap.layer6_ui.screens.session_start import SessionStartScreen
        from cap.layer6_ui.screens.scan_region import ScanRegionScreen
        from cap.layer6_ui.screens.scan_control import ScanControlScreen
        from cap.layer6_ui.screens.results_dashboard import ResultsDashboardScreen
        from cap.layer6_ui.screens.review_confirm import ReviewConfirmScreen
        from cap.layer6_ui.screens.patient_history import PatientHistoryScreen
        from cap.layer6_ui.screens.settings import SettingsScreen

        screen_classes = {
            SCREEN_SESSION_START: SessionStartScreen,
            SCREEN_SCAN_REGION: ScanRegionScreen,
            SCREEN_SCAN_CONTROL: ScanControlScreen,
            SCREEN_RESULTS: ResultsDashboardScreen,
            SCREEN_REVIEW: ReviewConfirmScreen,
            SCREEN_HISTORY: PatientHistoryScreen,
            SCREEN_SETTINGS: SettingsScreen,
        }

        for name, cls in screen_classes.items():
            screen = cls(
                app_context=self._ctx,
                nav_signals=self._nav_signals,
            )
            self._screens[name] = screen
            self._stack.addWidget(screen)

    def _create_toolbar(self) -> None:
        """Create the navigation toolbar."""
        toolbar = QToolBar("Navigation")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        # Navigation actions
        nav_items = [
            ("New Scan", SCREEN_SESSION_START),
            ("Scan Region", SCREEN_SCAN_REGION),
            ("Scan", SCREEN_SCAN_CONTROL),
            ("Results", SCREEN_RESULTS),
            ("Review", SCREEN_REVIEW),
            ("History", SCREEN_HISTORY),
            ("Settings", SCREEN_SETTINGS),
        ]

        for label, screen_name in nav_items:
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked, s=screen_name: self.navigate_to(s)
            )
            toolbar.addAction(action)

    def _setup_shortcuts(self) -> None:
        """Set up global keyboard shortcuts."""
        # Emergency stop: Escape key
        estop_action = QAction("Emergency Stop", self)
        estop_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        estop_action.triggered.connect(self._on_emergency_stop)
        self.addAction(estop_action)

    def navigate_to(self, screen_name: str) -> None:
        """
        Switch to a named screen.

        Parameters
        ----------
        screen_name : str
            One of the SCREEN_* constants.
        """
        if screen_name not in self._screens:
            logger.warning("Unknown screen: %s", screen_name)
            return

        screen = self._screens[screen_name]
        self._stack.setCurrentWidget(screen)

        # Notify the screen it's being shown
        if hasattr(screen, "on_shown"):
            screen.on_shown()

        self._status_bar.showMessage(f"Screen: {screen_name.replace('_', ' ').title()}")
        logger.debug("Navigated to: %s", screen_name)

    def _go_back(self) -> None:
        """Navigate to the previous screen in the workflow order."""
        current = self._stack.currentWidget()
        current_name = None
        for name, widget in self._screens.items():
            if widget is current:
                current_name = name
                break

        if current_name and current_name in SCREEN_ORDER:
            idx = SCREEN_ORDER.index(current_name)
            if idx > 0:
                self.navigate_to(SCREEN_ORDER[idx - 1])

    def _on_emergency_stop(self) -> None:
        """Handle emergency stop keyboard shortcut."""
        logger.warning("Emergency stop requested via keyboard")

        if hasattr(self._ctx, "motor"):
            self._ctx.motor.emergency_stop()

        self._status_bar.showMessage("EMERGENCY STOP — all motors halted")

        QMessageBox.warning(
            self,
            "Emergency Stop",
            "All motors have been halted.\n\n"
            "Check the microscope before resuming.\n"
            "The system will need to re-home before scanning.",
        )

    def closeEvent(self, event) -> None:
        """Handle window close — confirm and clean up."""
        reply = QMessageBox.question(
            self,
            "Exit CAP",
            "Are you sure you want to exit?\n\nAny scan in progress will be stopped.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Application exit requested")
            if hasattr(self._ctx, "shutdown"):
                self._ctx.shutdown()
            event.accept()
        else:
            event.ignore()


def launch_ui(app_context: AppContext) -> int:
    """
    Launch the CAP user interface.

    Parameters
    ----------
    app_context : AppContext
        The initialized application context from app.py.

    Returns
    -------
    int
        Application exit code.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("CAP")
    app.setOrganizationName("CAP")

    window = MainWindow(app_context)
    window.show()

    logger.info("UI launched")
    return app.exec()
