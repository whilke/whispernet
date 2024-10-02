using System.Net;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace WhisperNet
{
public class Inference
    {

        private static List<NamedOnnxValue> CreateInput(WhisperConfig config)
        {
            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("min_length", new DenseTensor<int>(new int[] {config.min_length}, new int[] { 1 })),
                NamedOnnxValue.CreateFromTensor("max_length", new DenseTensor<int>(new int[] {config.max_length}, new int[] { 1 })),
                NamedOnnxValue.CreateFromTensor("num_beams", new DenseTensor<int>(new int[] {config.num_beams}, new int[] { 1 })),
                NamedOnnxValue.CreateFromTensor("num_return_sequences", new DenseTensor<int>(new int[] {config.num_return_sequences}, new int[] { 1 })),
                NamedOnnxValue.CreateFromTensor("length_penalty", new DenseTensor<float>(new float[] {config.length_penalty}, new int[] { 1 })),
                NamedOnnxValue.CreateFromTensor("repetition_penalty", new DenseTensor<float>(new float[] {config.repetition_penalty}, new int[] { 1 })),
                NamedOnnxValue.CreateFromTensor("logits_processor", new DenseTensor<int>(new int[] {0}, new int[] { 1 })),
            };
            return input;
        }

        private List<NamedOnnxValue> model_inputs;
        private InferenceSession session;
        private WhisperConfig config;

        public Inference(WhisperConfig config, string modelPath)
        {
            this.config = config;

            var sessionOptions = config.GetSessionOptionsForEp();
            sessionOptions.RegisterOrtExtensions();

            session = new InferenceSession(modelPath, sessionOptions);
            this.model_inputs = CreateInput(config);
        }


        public string Run(byte[] data)
        {
            var audioRawData = new DenseTensor<byte>(data, new[] { 1, data.Length });

            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("audio_stream", audioRawData),
            };
            input.AddRange(model_inputs);

            // Run inference
            var run_options = new RunOptions();


            List<string> outputs = new List<string>() { "str" };
            var result = session.Run(input, outputs, run_options);

            var stringOutput = (result.ToList().First().Value as IEnumerable<string>).ToArray();
            return stringOutput[0];
        }

    }
}
