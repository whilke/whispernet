using System.Buffers.Binary;
using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics;
using NAudio.Wave;

namespace WhisperNet
{
    internal class Program
    {
        static async Task Main(string[] args)
        {

            var dictor = new Dictator("./onnx/model.onnx");

            List<WhisperRecord> completedRecords = new();


            dictor.Whisper += (record) =>
            {
                Console.WriteLine("Diction: ");

                if (record.IsComplete)
                {
                    completedRecords.Add(record);
                }


                foreach(var completed in completedRecords)
                {
                    Console.Write(completed.Text);
                }
                Console.WriteLine(record.Text);
            };

            Console.WriteLine("Press ENTER to start recording");
            Console.ReadLine();
            dictor.Start();


            Console.WriteLine("Press ENTER to stop recording");
            Console.ReadLine();
            dictor.Stop();
        }

    }
}
