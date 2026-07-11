"""
SystemResourceObserver — CPU / RAM / GPU / Battery Monitor
===========================================================
Polls system resource utilisation every N seconds and publishes:

  context.resource_update   — periodic snapshot (always)
  context.resource_alert    — when a value crosses a threshold (debounced)

Resource sources
----------------
  CPU:     psutil.cpu_percent(interval=None)  — non-blocking, uses cached value
  RAM:     psutil.virtual_memory()
  Battery: psutil.sensors_battery()
  Disk I/O: psutil.disk_io_counters()        — delta between polls
  GPU:     Attempted via torch.cuda (already a dep) then None fallback

Threshold alerts
----------------
Each resource has a threshold from config. Once an alert fires, it is
suppressed until the value drops back below (threshold - hysteresis).
This prevents alert storms when the value oscillates around the threshold.

  resource  default_threshold   hysteresis
  --------  ----------------   ----------
  cpu       90%                 5%
  ram       85%                 5%
  battery   20% (low alert)    5%
  gpu       95%                 5%
"""

from __future__ import annotations

import asyncio
import time

import psutil

from spidy.core.event_bus import EventBus
from spidy.logging.logger import get_logger
from spidy.perception.context.base import BaseObserver
from spidy.perception.context.events import ResourceAlertEvent, ResourceUpdateEvent
from spidy.perception.context.snapshot import ResourceInfo

log = get_logger(__name__)


class SystemResourceObserver(BaseObserver):
    """
    Polls system resources and fires update + alert events.

    Parameters
    ----------
    bus:
        Application EventBus.
    poll_interval:
        Seconds between resource polls. Default: 5.0s.
    cpu_alert_threshold:
        Fire alert when CPU > this %. Default: 90.
    ram_alert_threshold:
        Fire alert when RAM > this %. Default: 85.
    battery_alert_threshold:
        Fire alert when battery < this % (unplugged). Default: 20.
    gpu_alert_threshold:
        Fire alert when GPU > this %. Default: 95.
    """

    name = "system_resources"

    def __init__(
        self,
        bus: EventBus,
        poll_interval: float = 5.0,
        cpu_alert_threshold: float = 90.0,
        ram_alert_threshold: float = 85.0,
        battery_alert_threshold: float = 20.0,
        gpu_alert_threshold: float = 95.0,
    ) -> None:
        super().__init__(bus, poll_interval)
        self._cpu_threshold = cpu_alert_threshold
        self._ram_threshold = ram_alert_threshold
        self._bat_threshold = battery_alert_threshold
        self._gpu_threshold = gpu_alert_threshold
        self._hysteresis = 5.0

        # Alert state: True while the resource is above threshold
        self._alert_active: dict[str, bool] = {
            "cpu": False,
            "ram": False,
            "battery": False,
            "gpu": False,
        }

        # Disk I/O delta state
        self._last_disk_read: int = 0
        self._last_disk_write: int = 0
        self._last_disk_time: float = time.monotonic()

        # Latest resource snapshot (readable by ObserverManager)
        self._latest: ResourceInfo = ResourceInfo()

    async def on_start(self) -> None:
        # Warm up CPU percent (first call always returns 0.0)
        await asyncio.to_thread(lambda: psutil.cpu_percent(interval=None))
        # Capture initial disk counters
        counters = await asyncio.to_thread(self._get_disk_counters)
        if counters:
            self._last_disk_read = counters[0]
            self._last_disk_write = counters[1]
        log.debug("SystemResourceObserver started.")

    async def poll(self) -> None:
        info = await asyncio.to_thread(self._collect)
        self._latest = info

        await self._bus.publish(ResourceUpdateEvent(
            cpu_pct=info.cpu_pct,
            ram_pct=info.ram_pct,
            ram_used_gb=info.ram_used_gb,
            ram_total_gb=info.ram_total_gb,
            gpu_pct=info.gpu_pct,
            gpu_vram_used_gb=info.gpu_vram_used_gb,
            battery_pct=info.battery_pct,
            battery_plugged=info.battery_plugged,
            disk_read_mb_s=info.disk_read_mb_s,
            disk_write_mb_s=info.disk_write_mb_s,
        ))

        await self._check_alerts(info)

    @property
    def latest(self) -> ResourceInfo:
        """Last observed ResourceInfo (readable from ObserverManager)."""
        return self._latest

    # ── Internal ──────────────────────────────────────────────────────────

    def _collect(self) -> ResourceInfo:
        """Collect all resource metrics. Runs in a thread."""
        # CPU
        cpu_pct = psutil.cpu_percent(interval=None)

        # RAM
        mem = psutil.virtual_memory()
        ram_pct = mem.percent
        ram_used_gb = mem.used / (1024 ** 3)
        ram_total_gb = mem.total / (1024 ** 3)

        # Battery
        battery = psutil.sensors_battery()
        if battery is not None:
            battery_pct = battery.percent
            battery_plugged = battery.power_plugged
        else:
            battery_pct = -1.0
            battery_plugged = False

        # GPU (via torch if available)
        gpu_pct, gpu_vram_used = self._get_gpu_metrics()

        # Disk I/O delta
        disk_read_mb_s, disk_write_mb_s = self._get_disk_delta()

        return ResourceInfo(
            cpu_pct=round(cpu_pct, 1),
            ram_pct=round(ram_pct, 1),
            ram_used_gb=round(ram_used_gb, 2),
            ram_total_gb=round(ram_total_gb, 2),
            gpu_pct=round(gpu_pct, 1),
            gpu_vram_used_gb=round(gpu_vram_used, 2),
            battery_pct=round(battery_pct, 1),
            battery_plugged=battery_plugged,
            disk_read_mb_s=round(disk_read_mb_s, 2),
            disk_write_mb_s=round(disk_write_mb_s, 2),
        )

    def _get_gpu_metrics(self) -> tuple[float, float]:
        """Try to get GPU utilisation via torch.cuda. Returns (pct, vram_gb)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0, 0.0
            # torch doesn't expose utilisation % directly — use memory ratio
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            vram_used_gb = allocated / (1024 ** 3)
            gpu_pct = (reserved / total * 100.0) if total > 0 else 0.0
            return gpu_pct, vram_used_gb
        except Exception:  # noqa: BLE001
            return 0.0, 0.0

    @staticmethod
    def _get_disk_counters() -> tuple[int, int] | None:
        """Return (bytes_read, bytes_written) from psutil."""
        try:
            counters = psutil.disk_io_counters()
            if counters:
                return counters.read_bytes, counters.write_bytes
        except Exception:  # noqa: BLE001
            pass
        return None

    def _get_disk_delta(self) -> tuple[float, float]:
        """Compute disk MB/s since last call."""
        now = time.monotonic()
        elapsed = now - self._last_disk_time
        counters = self._get_disk_counters()
        if counters is None or elapsed <= 0:
            return 0.0, 0.0

        read_bytes, write_bytes = counters
        read_delta = max(0, read_bytes - self._last_disk_read)
        write_delta = max(0, write_bytes - self._last_disk_write)

        self._last_disk_read = read_bytes
        self._last_disk_write = write_bytes
        self._last_disk_time = now

        read_mb_s = read_delta / (1024 * 1024) / elapsed
        write_mb_s = write_delta / (1024 * 1024) / elapsed
        return read_mb_s, write_mb_s

    async def _check_alerts(self, info: ResourceInfo) -> None:
        """Fire ResourceAlertEvent if thresholds are crossed (with hysteresis)."""
        # CPU high alert
        await self._evaluate_alert(
            "cpu", info.cpu_pct, self._cpu_threshold,
            high_alert=True,
            message=f"CPU usage is high: {info.cpu_pct:.0f}%",
        )
        # RAM high alert
        await self._evaluate_alert(
            "ram", info.ram_pct, self._ram_threshold,
            high_alert=True,
            message=f"RAM usage is high: {info.ram_pct:.0f}%",
        )
        # Battery low alert (only when not plugged)
        if info.has_battery and not info.battery_plugged:
            await self._evaluate_alert(
                "battery", info.battery_pct, self._bat_threshold,
                high_alert=False,   # Low alert: fires when value drops BELOW threshold
                message=f"Battery is low: {info.battery_pct:.0f}%",
            )
        # GPU high alert
        if info.gpu_pct > 0:
            await self._evaluate_alert(
                "gpu", info.gpu_pct, self._gpu_threshold,
                high_alert=True,
                message=f"GPU usage is high: {info.gpu_pct:.0f}%",
            )

    async def _evaluate_alert(
        self,
        resource: str,
        value: float,
        threshold: float,
        high_alert: bool,
        message: str,
    ) -> None:
        """
        Evaluate and fire an alert with hysteresis.

        high_alert=True:  alert when value > threshold
        high_alert=False: alert when value < threshold (low battery)
        """
        is_over = value > threshold if high_alert else value < threshold
        clear_point = (threshold - self._hysteresis) if high_alert else (threshold + self._hysteresis)
        is_clear = value < clear_point if high_alert else value > clear_point

        was_alerting = self._alert_active[resource]

        if is_over and not was_alerting:
            self._alert_active[resource] = True
            log.warning("Resource alert: {msg}", msg=message)
            await self._bus.publish(ResourceAlertEvent(
                resource=resource,
                value=value,
                threshold=threshold,
                message=message,
            ))
        elif is_clear and was_alerting:
            self._alert_active[resource] = False
            log.info("Resource alert cleared: {resource}", resource=resource)
