"""Data Studio「超声差分图」：四通道静止参考差分 US 瀑布图。

把某一目标 Trial 的四通道超声 log 包络减去同一受试者独立静止参考的均值包络，
画成四幅纵向瀑布图（浅上深下，红 = 回波包络增强、蓝 = 减弱），并叠加激活/释放
起止四个事件标签。纯 NumPy + pyqtgraph，不依赖 OpenSim / scipy。
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from .local_tools import UsDiffWaterfallPlayback
from .plots import _PROMPT_NAME_PEN

# 图上只展示这四个语义事件，配色与全屏回放一致（激活红 / 释放青，start 实线 / end 虚线）。
_LABEL_DISPLAY = {
    "activate_start": "激活start",
    "activate_end": "激活end",
    "release_start": "释放start",
    "release_end": "释放end",
}
_LABEL_ORDER = ("activate_start", "activate_end", "release_start", "release_end")

# 发散色：蓝（回波包络减弱）→ 白（无变化）→ 红（回波包络增强）。
_DIVERGING = pg.ColorMap(
    np.array([0.0, 0.5, 1.0]),
    np.array(
        [
            [33, 102, 172, 255],   # #2166ac
            [247, 247, 247, 255],  # #f7f7f7
            [178, 24, 43, 255],    # #b2182b
        ],
        dtype=np.ubyte,
    ),
)

# 深度轴：ADC 20 在上（浅）、976 在下（深），对应 957 个深度点。
_DEPTH_TOP = 19.5
_DEPTH_BOTTOM = 976.5


def _label_pen(name: str) -> pg.mkPen:
    color, style = _PROMPT_NAME_PEN.get(name, ("#ff0000", Qt.PenStyle.SolidLine))
    return pg.mkPen(color, width=1.8, style=style)


class UsDiffPreviewWindow(QMainWindow):
    """静态四通道差分瀑布图窗口，接受已算好的 :class:`UsDiffWaterfallPlayback`。"""

    def __init__(
        self,
        playback: UsDiffWaterfallPlayback,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle("超声差分图 · 回波包络差（当前 − 静止参考）")
        self.resize(1200, 880)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QLabel(
            "四通道超声<b>回波包络差</b>瀑布图（当前 − 静止参考，<b>非肌力大小</b>）："
            "红色 = 包络增强，蓝色 = 包络减弱；深度浅在上（ADC 20）、深在下（ADC 976）。"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        info = QLabel(
            f"目标 {playback.target_condition_code} · 静止参考 "
            f"{playback.reference_condition_code}（参考段 5–25 s，"
            f"时长 {playback.reference_duration_s:.3f} s / "
            f"{playback.reference_interval_valid_frames} 有效帧） · "
            f"共同色轴 ±{playback.color_limit:.4g}（|差分|99%）"
        )
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { color: #46566b; }")
        layout.addWidget(info)

        self._graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graphics, 1)

        time = playback.time_s
        x0 = float(time[0]) - 0.025 if time.size else 0.0
        x1 = float(time[-1]) + 0.025 if time.size else 1.0
        span = max(x1 - x0, 1e-6)

        plots: list[pg.PlotItem] = []
        for channel in range(4):
            plot = self._graphics.addPlot(row=channel, col=0)
            plot.setTitle(f"通道 {channel + 1}", color="#000000", size="10pt")
            plot.setLabel("left", "深度 ADC")
            plot.showGrid(x=True, y=False, alpha=0.2)
            plot.getAxis("left").setWidth(58)
            if channel == 3:
                plot.setLabel("bottom", "相对正式采集起点时间", units="s")
            else:
                plot.getAxis("bottom").setStyle(showValues=False)
            if plots:
                plot.getViewBox().setXLink(plots[0].getViewBox())

            # 无效帧（未对齐/非有限）显示为 0（中性白），避免被误读成强减弱。
            image_data = np.nan_to_num(
                playback.diff[:, channel, :].T, nan=0.0, posinf=0.0, neginf=0.0
            )
            image = pg.ImageItem(axisOrder="row-major")
            image.setImage(image_data)
            image.setLevels([-playback.color_limit, playback.color_limit])
            image.setColorMap(_DIVERGING)
            image.setRect(
                QRectF(x0, _DEPTH_TOP, span, _DEPTH_BOTTOM - _DEPTH_TOP)
            )
            plot.addItem(image)
            plot.setYRange(_DEPTH_BOTTOM, _DEPTH_TOP, padding=0.0)
            plot.setXRange(x0, x1, padding=0.0)

            for event in playback.labels:
                line = pg.InfiniteLine(
                    pos=float(event.time_s),
                    angle=90,
                    movable=False,
                    pen=_label_pen(event.name or ""),
                )
                line.setZValue(90)
                line.setToolTip(
                    f"{_LABEL_DISPLAY.get(event.name or '', event.name)} · "
                    f"t={event.time_s:.3f} s"
                )
                plot.addItem(line)
            plots.append(plot)

        bar = pg.ColorBarItem(
            values=(-playback.color_limit, playback.color_limit),
            colorMap=_DIVERGING,
            width=16,
            label="log1p 包络差（当前 − 静止参考）",
        )
        self._graphics.addItem(bar, row=0, col=1, rowspan=4)

        legend = QHBoxLayout()
        legend.setContentsMargins(0, 0, 0, 0)
        legend.addWidget(QLabel("事件标签："))
        for name in _LABEL_ORDER:
            color, style = _PROMPT_NAME_PEN.get(
                name, ("#ff0000", Qt.PenStyle.SolidLine)
            )
            marker = "┈┈" if style == Qt.PenStyle.DashLine else "━━"
            legend.addWidget(
                QLabel(f'<span style="color:{color};">{marker} {_LABEL_DISPLAY[name]}</span>')
            )
        legend.addStretch(1)
        layout.addLayout(legend)

        self.setCentralWidget(central)


__all__ = ["UsDiffPreviewWindow"]
