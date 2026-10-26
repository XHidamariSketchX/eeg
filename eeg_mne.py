"""用 MNE 分析本项目录制 CSV；默认不假定输出采样率。"""
import argparse
import csv
from datetime import datetime
import html
import json
from pathlib import Path

import numpy as np

COLUMNS = ['packet_counter', 'wear_detection', 'ADC1_raw', 'ADC2_raw',
           'ADC1_filtered', 'ADC2_filtered', 'theta', 'alpha', 'beta',
           'attention', 'relaxation', 'host_monotonic_seconds']


def load_recording(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'缺少录制字段：{sorted(missing)}')
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError('至少需要两行数据')
    data = {}
    for key in COLUMNS:
        data[key] = np.array([float(row[key]) for row in rows])
        if not np.isfinite(data[key]).all():
            raise ValueError(f'{key} 包含无效数值')
    times = data['host_monotonic_seconds']
    if np.any(np.diff(times) < 0):
        raise ValueError('主机单调时间发生倒退')
    counters = data['packet_counter']
    if np.any(counters != np.floor(counters)) or np.any((counters < 0) | (counters > 255)):
        raise ValueError('包序号必须为 0–255 的整数')
    return data


def quality(data, wear_ok=1):
    """连续段只按佩戴、序号、接收长间隔筛选，不声称已去除动作伪迹。"""
    times = data['host_monotonic_seconds']
    bad = data['wear_detection'] != wear_ok
    boundaries = np.r_[False, (np.diff(data['packet_counter']) % 256 != 1) | (np.diff(times) > 1)]
    segments = []
    start = None
    for index in range(len(times)):
        if bad[index] or boundaries[index]:
            if start is not None:
                segments.append((start, index))
                start = None
        if not bad[index] and start is None:
            start = index
    if start is not None:
        segments.append((start, len(times)))
    return bad, boundaries, segments


def compute_spectrum(data, segments, fs=None, source='filtered'):
    from mne.time_frequency import psd_array_welch
    # 固定 128 点段长；只使用未跨越状态/序号/长间隔边界的连续片段。
    nperseg = 128
    estimates = []
    for start, end in segments:
        if end - start < nperseg:
            continue
        x = np.vstack([data[f'ADC{i}_{source}'][start:end] for i in (1, 2)])
        psd, frequencies = psd_array_welch(
            x, sfreq=fs if fs is not None else 1.0, n_fft=nperseg,
            n_per_seg=nperseg, n_overlap=64, average=None, verbose=False)
        estimates.append(psd)
    if not estimates:
        return None
    windows = np.concatenate(estimates, axis=-1)
    return frequencies, windows.mean(axis=-1), windows.shape[-1]


def make_raw(data, fs, wear_ok=1):
    import mne
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('需要有限的正数输出采样率假设')
    names = ['ADC1_filtered', 'ADC2_filtered', 'theta', 'alpha', 'beta', 'attention', 'relaxation']
    # 尚无电压换算依据，使用 misc，避免 MNE 按伏或微伏解释设备输出数值。
    raw = mne.io.RawArray(np.vstack([data[name] for name in names]),
                         mne.create_info(names, sfreq=fs, ch_types=['misc'] * len(names)), verbose=False)
    raw.info['description'] = f'Output sampling rate ASSUMED {fs:g} Hz; values uncalibrated; device bands in percent.'
    bad, boundaries, _ = quality(data, wear_ok)
    annotations = []
    padded = np.r_[False, bad, False].astype(int)
    starts, ends = np.where(np.diff(padded) == 1)[0], np.where(np.diff(padded) == -1)[0]
    for start, end in zip(starts, ends):
        annotations.append((start/fs, (end-start)/fs, 'BAD_wear'))
    for index in np.flatnonzero(boundaries):
        annotations.append((index/fs, 1/fs, 'BAD_boundary'))
    if annotations:
        raw.set_annotations(mne.Annotations(*zip(*annotations)))
    return raw


def run(args):
    import matplotlib
    if not args.browse:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mne
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    path = args.csv
    data = load_recording(path)
    bad, boundaries, segments = quality(data, args.wear_ok)
    times = data['host_monotonic_seconds'] - data['host_monotonic_seconds'][0]
    size = len(times)
    rate = (size-1)/times[-1] if times[-1] > 0 else None
    counter_jumps = int(np.count_nonzero(np.diff(data['packet_counter']) % 256 != 1))
    result = compute_spectrum(data, segments, args.fs, args.source)
    output = Path(__file__).resolve().parent / 'analysis' / f'{path.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True)
    summary = {
        'source_csv': str(path.resolve()), 'mne_version': mne.__version__,
        'rows': size, 'host_receive_span_seconds': float(times[-1]), 'host_receive_rows_per_second': rate,
        'assumed_output_sampling_rate_hz': args.fs, 'source_channels': args.source,
        'wear_ok_value': args.wear_ok, 'wear_bad_rows': int(bad.sum()),
        'counter_discontinuities': counter_jumps, 'receive_gaps_over_1s': int((np.diff(times)>1).sum()),
        'spectrum_windows': result[2] if result else 0,
        'voltage_calibration': 'unknown', 'artifact_rejection': 'wear, counter and gaps only; movement not automatically removed',
    }
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report = mne.Report(title='双通道设备 · MNE 分析', verbose=False)
    basis = f'指定输出采样率假设：{args.fs:g} Hz（未验证）' if args.fs else '输出采样率未知；频谱使用 cycles/sample，不解释为 Hz'
    rows = ''.join(f'<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>' for k,v in summary.items())
    report.add_html(f'<p>{basis}</p><p>接收时间用于传输统计，不能当作设备采样时间。原始 ADC 列是厂家命名，不保证保留全部硬件采样点。质量筛选不等于去除运动伪迹。</p><table>{rows}</table>', title='数据与分析条件')

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
    sample_index = np.arange(size)
    for i, ax in enumerate(axes, 1):
        ax.plot(sample_index, data[f'ADC{i}_{args.source}'], lw=.7, label=f'ADC{i}_{args.source}')
        if bad.any():
            ax.scatter(sample_index[bad], data[f'ADC{i}_{args.source}'][bad], color='red', s=3, label='佩戴状态异常')
        for boundary in np.flatnonzero(boundaries):
            ax.axvline(boundary, color='orange', alpha=.4)
        ax.set_ylabel('设备输出值（单位未标定）')
        ax.legend(loc='upper right')
        ax.grid(alpha=.25)
    axes[-1].set_xlabel('录制样本序号（不是秒）')
    fig.suptitle('双通道输出序列 · 红点为佩戴异常，橙线为连续性边界')
    fig.savefig(output/'waveforms.png', dpi=130)
    report.add_figure(fig, title='双通道序列')
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout='constrained')
    for name in ['theta','alpha','beta']:
        axes[0].plot(times, data[name], label=name)
    axes[0].set_ylabel('设备波段占比 (%)')
    for name in ['attention','relaxation']:
        axes[1].plot(times, data[name], label=name)
    axes[1].set_ylabel('设备评分')
    axes[2].plot(times[1:], np.diff(times)*1000, lw=.8, label='相邻行接收间隔')
    axes[2].set_ylabel('毫秒')
    axes[2].set_xlabel('主机接收经过时间（秒）')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc='upper right')
    fig.savefig(output/'device_metrics.png', dpi=130)
    report.add_figure(fig, title='设备指标与传输')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    if result:
        frequencies, psd, window_count = result
        for i in range(2):
            ax.plot(frequencies, psd[i], label=f'ADC{i+1}')
        np.savetxt(output/'spectrum.csv', np.column_stack([frequencies,psd.T]), delimiter=',',
                   header=('frequency_hz_assumed' if args.fs else 'cycles_per_sample')+',ADC1_psd,ADC2_psd',comments='')
        ax.legend()
    else:
        ax.text(.5,.5,'没有至少 128 点的有效连续片段，未计算频谱',ha='center',transform=ax.transAxes)
    ax.set_title('MNE Welch · '+basis)
    ax.set_xlabel('频率 (Hz，假设)' if args.fs else '归一化频率 (cycles/sample)')
    ax.set_ylabel('输出值²/Hz' if args.fs else '输出值²/(cycles/sample)')
    ax.grid(alpha=.25)
    fig.savefig(output/'spectrum.png', dpi=130)
    report.add_figure(fig, title='输出序列频谱')
    plt.close(fig)
    report.save(output/'report.html', open_browser=False, overwrite=False, verbose=False)
    print(f'记录 {size} 行；主机接收跨度 {times[-1]:.3f} 秒；平均速率 {rate if rate is not None else "不可估计"} 行/秒')
    print(f'报告：{output / "report.html"}')
    if args.fs is not None:
        raw = make_raw(data, args.fs, args.wear_ok)
        raw.save(output/'assumed_rate_raw.fif', overwrite=False, verbose=False)
        if args.browse:
            mne.viz.set_browser_backend('matplotlib')
            raw.plot(scalings='auto', duration=min(10,raw.times[-1]), block=True,
                     title=f'输出序列 · 假设 {args.fs:g} Hz · misc 通道未做电压标定')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',nargs='?',type=Path,help='不指定时读取 recordings 中最新 eeg_*.csv')
    parser.add_argument('--fs',type=float,help='显式指定输出采样率假设；省略则只算归一化频谱')
    parser.add_argument('--source',choices=['raw','filtered'],default='filtered')
    parser.add_argument('--wear-ok',type=int,default=1)
    parser.add_argument('--browse',action='store_true',help='生成报告后打开 MNE 波形浏览器，需要 --fs')
    args=parser.parse_args()
    if args.fs is not None and (not np.isfinite(args.fs) or args.fs<=0):
        parser.error('--fs 必须为有限正数')
    if args.browse and args.fs is None:
        parser.error('--browse 需要显式指定 --fs，浏览器时间轴依赖该假设')
    if args.csv is None:
        files=list((Path(__file__).resolve().parent/'recordings').glob('eeg_*.csv'))
        if not files:
            parser.error('没有录制文件，请先用 eeg_viewer.py 录制')
        args.csv=max(files,key=lambda p:p.stat().st_mtime)
    try:
        run(args)
    except (OSError,ValueError) as exc:
        parser.exit(1,f'分析失败：{exc}\n')


if __name__=='__main__':
    main()
