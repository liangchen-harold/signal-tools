"""
麦克风频域示波器，通过瀑布图显示. Plot the live microphone signal(s) in waterfall.
"""
import argparse
import queue
import sys
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import scipy

q = queue.Queue()

# 实信号转为复信号
def to_complex(data):
    signal_mean = data.mean()
    signal_pp = data.max() - data.min() + 1e-9
    samples = (data - signal_mean) / signal_pp * 2

    # 标准方法：希尔伯特变换
    samples = scipy.signal.hilbert(samples)
    return samples

# 频域搬移
def convert_down(samples, freq, sample_rate):
    Ts = len(samples)/sample_rate
    # time vector
    t = np.linspace(0, Ts, len(samples))
    samples = samples * np.exp(-1j*2*np.pi*freq*t)
    return samples

# 带通滤波器
from scipy.signal import butter, lfilter, freqz
def butter_bandpass(center_freq, half_band, fs, order=5):
    return butter(order, [center_freq-half_band, center_freq+half_band], fs=fs, btype='band')
def butter_bandpass_filter(data, center_freq, half_band, fs, order=5):
    b, a = butter_bandpass(center_freq, half_band, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 傅立叶变换（时域信号转换到频域）
def fft(samples, sample_rate):
    psd = np.fft.fftshift(np.abs(np.fft.fft(samples)))
    f = np.linspace(-sample_rate/2.0, sample_rate/2.0, len(psd))
    max_freq = f[np.argmax(psd)]
    return f, psd

# 录音回调（从工作线程调用），不能直接操作UI，需要通过queue把数据传递给主线程后处理和显示
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata[::, 0].copy())

# 更新UI绘图
def update_plot(plotdata, waterfall, ax, samplerate, negative):
    i = 0
    while True:
        try:
            data = q.get_nowait()
            i += data.shape[0]
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:] = data[:len(plotdata)]

    samples = plotdata
    # 转成复信号
    samples = to_complex(samples)
    # 滤波（可选）
    # samples = butter_bandpass_filter(samples, 2000, 500, samplerate, order = 6)
    # 频域搬移（可选）
    # samples = convert_down(samples, 2000, samplerate)

    # 转频域
    f, data = fft(samples, samplerate)
    # 频域数据记录到历史
    waterfall[1:, :] = waterfall[0:-1, :]
    if negative:
        waterfall[0, :] = data[:]
    else:
        waterfall[0, :] = data[len(data)//2:]
    # print(plotdata.shape, i)
    if negative:
        fmin = np.min(f)
    else:
        fmin = 0
    fmax = np.max(f)
    ax.clear()

    mode = 'waterfall'
    # mode = 'plot'
    if mode == 'waterfall':
        # vmax = 0.1
        vmax = np.quantile(waterfall, 0.999)
        # print(vmax)
        ax.imshow(waterfall,cmap='inferno',vmax=vmax,extent=[fmin,fmax,0,1],interpolation='nearest',origin='upper',aspect='auto')
        major_ticks = np.arange(fmin, fmax, 1000)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.get_yaxis().set_visible(False)
        # ax.invert_yaxis()
    else:
        ax.plot(f, data)
        major_ticks = np.arange(fmin,fmax, 1000)
        # minor_ticks = np.arange(fmin,fmax, 100)
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlim([0, np.max(f)])
        # ax.set_xticks(minor_ticks, minor=True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%6.1f'))
    ax.grid(which='both', axis='x')


def waterfall(samplerate, negative=False, device=None):
    print(sd.query_devices())

    # 初始化变量
    length = 2048
    plotdata = np.zeros((length))
    waterfall = np.zeros((256, length if negative else length//2))

    # 录音
    stream = sd.InputStream(device=device, channels=1, samplerate=samplerate, callback=audio_callback)
    stream.start()

    # 初始化绘图
    fig, ax = plt.subplots(figsize=(8, 5))

    # 检测是不是从jupyter运行
    is_jupyter = False
    try:
        display
    except NameError:
        pass
    else:
        from IPython.display import clear_output
        is_jupyter = True

    try:
        # 主循环
        while True:
            update_plot(plotdata, waterfall, ax, samplerate, negative)
            fig.tight_layout(pad=0)

            if is_jupyter:
                clear_output(wait = True)
                display(fig)
            else:
                plt.draw()
                plt.pause(0.001)

    except KeyboardInterrupt as e:
        stream.stop()

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', type=int, help='input device (numeric ID)')
    parser.add_argument('-r', '--samplerate', default=22000, type=float, help='sampling rate of audio device')
    parser.add_argument('-n', '--negative', default=False, type=bool, help='show negative')
    args, _ = parser.parse_known_args()

    waterfall(args.samplerate, args.negative, args.device)
