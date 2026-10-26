import math
import time

import serial
from pylsl import StreamInfo, StreamOutlet

SERIAL_PORT = "COM3"  # 与串口调试工具一致
BAUD_RATE = 115200
EXPECTED_COLUMNS = 21
# 列号从 1 开始：(7, 8) 为设备滤波输出，(5, 6) 为原始 ADC。
# None 表示广播全部 21 列。
EEG_COLUMNS = (7, 8)
HARDWARE_SAMPLE_RATE = 250  # 厂家文档：硬件 ADC 采样率
SAMPLE_RATE = 0  # CSV 每秒输出行数尚未确认；不能直接等同于硬件采样率
FIELD_LABELS = (
    "packet_counter", "battery_voltage", "wear_detection", "wear_detection_value",
    "ADC1_raw", "ADC2_raw", "ADC1_filtered", "ADC2_filtered",
    "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z",
    "delta", "theta", "alpha", "beta", "gamma", "attention", "relaxation",
)


def parse_line(raw):
    fields = raw.decode("ascii").strip().split(",")
    if len(fields) != EXPECTED_COLUMNS:
        raise ValueError(f"期望 {EXPECTED_COLUMNS} 列，收到 {len(fields)} 列")
    values = [float(field) for field in fields]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("数据包含非有限数值")
    return values


def main():
    selected = tuple(EEG_COLUMNS) if EEG_COLUMNS is not None else tuple(range(1, EXPECTED_COLUMNS + 1))
    if EEG_COLUMNS is not None and (
        len(selected) != 2 or len(set(selected)) != 2
        or any(not isinstance(col, int) or not 1 <= col <= EXPECTED_COLUMNS for col in selected)
    ):
        raise ValueError("EEG_COLUMNS 必须是两个不同的有效列号（从 1 开始）")
    name = "Gemini-EEG" if EEG_COLUMNS is not None else "Gemini-Serial"
    kind = "EEG" if EEG_COLUMNS is not None else "Data"
    info = StreamInfo(name, kind, len(selected), SAMPLE_RATE, "double64",
                      f"gemini_csv_{SERIAL_PORT}_" + "_".join(map(str, selected)))
    channels = info.desc().append_child("channels")
    info.desc().append_child_value("hardware_sampling_rate_hz", str(HARDWARE_SAMPLE_RATE))
    for col in selected:
        channel = channels.append_child("channel")
        channel.append_child_value("label", FIELD_LABELS[col - 1])
        channel.append_child_value("csv_column", str(col))
    if selected == (7, 8):
        info.desc().append_child_value(
            "filtering", "Device Butterworth IIR bandpass 0.5-30 Hz; 8th order, 4 SOS (vendor specification)"
        )
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.2) as ser:
            outlet = StreamOutlet(info)
            print(f"已打开 {SERIAL_PORT}，波特率 {BAUD_RATE}；LSL 流：{name}", flush=True)
            print(f"广播列号：{selected}；等待 CSV 数据…", flush=True)
            print(f"通道标签：{[FIELD_LABELS[col - 1] for col in selected]}", flush=True)
            pending = bytearray()
            received = skipped = 0
            last_report = time.monotonic()
            last_received = 0
            while True:
                # 超时收到的半行保留到下一次读取。
                pending.extend(ser.read(ser.in_waiting or 1))
                while b"\n" in pending:
                    raw, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    if not raw.strip():
                        continue
                    try:
                        values = parse_line(raw)
                    except (UnicodeDecodeError, ValueError) as exc:
                        skipped += 1
                        if skipped <= 3:
                            print(f"跳过无效行：{exc}；内容：{raw[:160]!r}", flush=True)
                        continue
                    outlet.push_sample([values[col - 1] for col in selected])
                    received += 1
                    if received == 1:
                        print(f"收到首行：{values}", flush=True)
                if len(pending) > 65536:
                    pending.clear()
                    skipped += 1
                    print("未检测到换行符，已清理过长数据；请核对串口配置。", flush=True)
                now = time.monotonic()
                if now - last_report >= 2:
                    rate = (received - last_received) / (now - last_report)
                    print(f"已转发 {received} 行，跳过 {skipped} 行；接收速率 {rate:.1f} 行/秒", flush=True)
                    last_received = received
                    last_report = now
    except serial.SerialException as exc:
        print(f"串口错误：{exc}\n请核对端口和波特率，并关闭占用串口的调试工具。", flush=True)
    except KeyboardInterrupt:
        print("已停止广播，串口已关闭。", flush=True)


if __name__ == "__main__":
    main()
