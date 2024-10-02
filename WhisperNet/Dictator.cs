using System.Buffers.Binary;
using NAudio.Wave;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace WhisperNet;

public class Dictator
{

    public delegate void OnWhisper(WhisperRecord record);

    private readonly Inference inference;

    private readonly int minDurationSeconds;
    private readonly int silentDuration;
    private readonly int silentDecThreshold;

    private readonly object lockObject = new(); 
    private readonly List<byte[]> audioData = [];

    private WaveInEvent waveIn = null!;
    private CancellationTokenSource tokenSource = null!;
    private Task audioTask = null!;

    public event OnWhisper Whisper = null!;
    
    public bool IsRecording { get; private set; } = false;
    public Dictator(string modelPath, int minDurationSeconds =  1, int silentDuration = 2, int silentDecThreshold = -40)
    {
        this.minDurationSeconds = minDurationSeconds;
        this.silentDuration = silentDuration;
        this.silentDecThreshold = silentDecThreshold;

        var config = new WhisperConfig();
        this.inference = new Inference(config, modelPath);
    }

    public void Start()
    {
        if (this.IsRecording) return;

        tokenSource?.Cancel();

        this.IsRecording = true;

        lock (lockObject)
        {
            audioData.Clear();
        }
        waveIn = new WaveInEvent();
        waveIn.WaveFormat = new WaveFormat(44100, 16, 1);
        waveIn.StartRecording();
        tokenSource = new CancellationTokenSource();
        audioTask = ProcessAudio(tokenSource.Token);

        this.SetupMicData();
    }

    public void Stop()
    {
        if (!this.IsRecording) return;

        this.IsRecording = false;
        waveIn.StopRecording();

        tokenSource?.Cancel();

    }

    private void SetupMicData()
    {
        waveIn.DataAvailable += (s, a) =>
        {
            var data = new byte[a.BytesRecorded];
            Array.Copy(a.Buffer, data, a.BytesRecorded);
            lock (lockObject)
            {
                audioData.Add(data);
            }

        };
    }

    private async Task ProcessAudio(CancellationToken token)
    {
        await Task.Yield();
        while (!token.IsCancellationRequested)
        {
            byte[] audio;
            lock (lockObject) audio = audioData.SelectMany(x => x).ToArray();

            var totalBuffer = audio.Length;
            var samples = totalBuffer / waveIn.WaveFormat.BlockAlign;
            var duration = samples / (float)waveIn.WaveFormat.SampleRate;

            if (duration >= this.minDurationSeconds)
            {
                var wasSilent = IsAudioSilent(audio, this.silentDecThreshold, this.silentDuration);
                if (duration <= this.silentDuration && wasSilent)
                {
                    continue;
                }

                var result = EncodeAudio(audio);
                result = result.Replace("[BLANK_AUDIO]", "");
                if (string.IsNullOrEmpty(result))
                {
                    continue;
                }

                if (duration >= 28 || wasSilent)
                {
                    Whisper?.Invoke(new WhisperRecord
                    {
                        Time = DateTime.Now,
                        Text = result,
                        IsComplete = true
                    });

                    //get last 1 second of audio
                    var lastSecond = audio.Skip(audio.Length - (waveIn.WaveFormat.SampleRate * waveIn.WaveFormat.BlockAlign)).ToArray();
                    lock (lockObject)
                    {
                        audioData.Clear();
                        audioData.Add(lastSecond);
                    }
                }
                else
                {
                    Whisper?.Invoke(new WhisperRecord
                    {
                        Time = DateTime.Now,
                        Text = result,
                        IsComplete = false
                    });
                }
            }

        }
    }

    private string EncodeAudio(byte[] data)
    {
        var memoryBuffer = new MemoryStream();
        var writer = new WaveFileWriter(memoryBuffer, waveIn.WaveFormat);
        writer.Write(data, 0, data.Length);
        writer.Dispose();
        var audio = memoryBuffer.ToArray();
        var result = inference.Run(audio);
        return result;
    }

    private bool IsAudioSilent(byte[] data, int threshold, int duration)
    {
        var bytePerSample = waveIn.WaveFormat.BitsPerSample / 8;
        float floatThreshold = (float)(Math.Pow(10.0, threshold / 20.0) * Math.Pow(2, bytePerSample * 8));

        var totalDuration = data.Length / (waveIn.WaveFormat.SampleRate * waveIn.WaveFormat.BlockAlign);

        byte[] audio = totalDuration <= duration ? data :
            data.Skip(data.Length - (waveIn.WaveFormat.SampleRate * waveIn.WaveFormat.BlockAlign * duration)).ToArray();

        int counterStart = -1;
        int counterLength = 0;

        for (int n = 0; n < audio.Length / bytePerSample; n++)
        {

            float toCheck = 0;
            // Little endian conversion
            if (bytePerSample == 1)
            {
                toCheck = audio[n] - 128;
            }
            else if (bytePerSample == 2)
            {
                Span<byte> toConvert = new byte[2] { audio[n * bytePerSample], audio[n * bytePerSample + 1] };
                toCheck = BinaryPrimitives.ReadInt16LittleEndian(toConvert);
            }
            else if (bytePerSample == 4)
            {
                Span<byte> toConvert = new byte[4] {
                    audio[n * bytePerSample],
                    audio[n * bytePerSample + 1],
                    audio[n * bytePerSample + 2],
                    audio[n * bytePerSample + 3]
                };
                toCheck = BinaryPrimitives.ReadInt32LittleEndian(toConvert);
            }

            if (Math.Abs(toCheck) > floatThreshold)
            {
                return false;
            }
        }
        return true;
    }

}